use anchor_lang::prelude::*;
use anchor_lang::system_program;
use anchor_lang::solana_program::{
    program::{invoke, invoke_signed},
    system_instruction,
    sysvar::{self, clock, instructions, rent},
};
use anchor_spl::associated_token::AssociatedToken;
use anchor_spl::token::{self, mint_to, MintTo, Token, TokenAccount, Mint};
use mpl_token_metadata::instruction::{
    create_metadata_accounts_v3,
    create_master_edition_v3
};
use mpl_token_metadata::state::Creator;
use std::collections::{HashMap, VecDeque};

use pyth_sdk_solana::state::load_price_account; // Using the new Pyth SDK

/// The program ID for this contract
declare_id!("GL2qB66X6cNEBB7y5U7C4R6kEtG3brQs3MVZ6GaaS9Cy");

// ====================================================
// CONSTANTS & PLACEHOLDERS
// ====================================================

pub const TERM_END_TIMESTAMP: i64 = 1_900_000_000;
pub const GLITCH_AUCTION_DATE_2026: i64 = 1_750_000_000;
pub const GLITCH_AUCTION_DATE_2027: i64 = 1_780_000_000;
pub const GLITCH_AUCTION_DATE_2028: i64 = 1_810_000_000;
pub const GLITCH_AUCTION_DATE_2029: i64 = 1_840_000_000;

/// The Metaplex Token Metadata program ID
pub const MPL_TOKEN_METADATA_ID: Pubkey = mpl_token_metadata::ID;

/// For reading Pyth data (the program ID for Pyth on Solana)
pub const PYTH_PROGRAM_ID: Pubkey = Pubkey::new_from_array([
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2,
]);

// ====================================================
// HELPER FUNCTIONS & UTILITIES
// ====================================================

/// Example integer sqrt used in volatility calc
fn integer_sqrt(value: u128) -> u64 {
    if value == 0 {
        return 0;
    }
    let mut x = value;
    let mut y = (x + 1) >> 1;
    while y < x {
        x = y;
        y = (x + value / x) >> 1;
    }
    x as u64
}

/// Absolute difference
fn abs_diff(a: u64, b: u64) -> u64 {
    if a > b { a - b } else { b - a }
}

// ====================================================
// EPHEMERAL STRUCTS (NO ANCHOR DERIVES)
// ====================================================

#[derive(Clone, Debug)]
pub enum PriceStatus {
    Trading,
    Halted,
    Unknown,
}

/// A simple struct for returning an oracle price
#[derive(Clone, Debug)]
pub struct OraclePrice {
    pub price: i64,
    pub confidence: u64,
    pub timestamp: i64,
    pub status: PriceStatus,
}

/// Distinguish Price feeds vs. Sentiment feeds
#[derive(Clone, AnchorSerialize, AnchorDeserialize)]
pub enum FeedKind {
    Price,
    Sentiment,
}

/// An oracle feed definition in EnhancedOracleAggregator
#[derive(Clone, AnchorSerialize, AnchorDeserialize)]
pub struct OracleFeed {
    pub feed_pubkey: Pubkey,
    pub previous_price_i64: i64,
    pub previous_confidence: u64,
    pub kind: FeedKind,
}

// ====================================================
// DATA STRUCTURES
// ====================================================

/// An account storing user fraction info
#[account]
pub struct FractionAccount {
    pub owner: Pubkey,
    pub total_contributed: u64,
    pub last_update_timestamp: i64,
    pub is_active: bool,
}
impl FractionAccount {
    pub const MAX_SIZE: usize = 8 + 32 + 8 + 8 + 1;
}

/// A single user fraction entry (bull or bear) with day, quantity, etc.
#[derive(Clone, AnchorSerialize, AnchorDeserialize)]
pub struct FractionEntry {
    pub day: u64,
    pub quantity: u64,
    pub initial_value: u64,
    pub is_bull: bool,
}

/// A user position with multiple fraction entries
#[derive(Clone, AnchorSerialize, AnchorDeserialize)]
pub struct UserPosition {
    pub last_consolidated_day: u64,
    pub consolidated_value: u64,
    pub raw_total_fractions: u64,
    pub recent_entries: VecDeque<FractionEntry>,
}

impl UserPosition {
    /// Consolidates fraction entries older than 10 days
    pub fn consolidate_if_needed(&mut self, current_day: u64) {
        let mut sum: u64 = 0;
        while let Some(entry) = self.recent_entries.front() {
            if current_day.saturating_sub(entry.day) >= 10 {
                let decayed = calculate_decayed_fraction_value(entry, current_day);
                sum = sum.saturating_add(decayed);
                self.recent_entries.pop_front();
            } else {
                break;
            }
        }
        self.consolidated_value = self.consolidated_value.saturating_add(sum);
    }

    /// Current total value (including consolidated + non-consolidated)
    pub fn total_value(&self, current_day: u64) -> u64 {
        let recent_sum: u64 = self.recent_entries
            .iter()
            .map(|entry| calculate_decayed_fraction_value(entry, current_day))
            .sum();
        self.consolidated_value.saturating_add(recent_sum)
    }

    /// Resets everything
    pub fn reset(&mut self) {
        self.consolidated_value = 0;
        self.raw_total_fractions = 0;
        self.recent_entries.clear();
    }
}

/// Config for daily decay rates
#[derive(Clone, AnchorSerialize, AnchorDeserialize)]
pub struct DecayRates {
    pub winning_side_daily_bps: u64,
    pub losing_side_daily_bps: u64,
}

/// A checkpoint for yearly aggregation
#[derive(Clone, AnchorSerialize, AnchorDeserialize)]
pub struct YearCheckpoint {
    pub year: u64,
    pub total_bull_value: u64,
    pub total_bear_value: u64,
    pub consolidated_day: u64,
}

/// Info for a “price pusher” (someone allowed to push oracle updates)
#[derive(Clone, AnchorSerialize, AnchorDeserialize)]
pub struct PricePusherInfo {
    pub last_push: i64,
    pub push_count: u64,
    pub is_active: bool,
}

/// Price config
#[derive(Clone, AnchorSerialize, AnchorDeserialize, Default)]
pub struct PriceConfig {
    pub weighting_spy: u64,
    pub weighting_btc: u64,
    pub volume_weight: u64,
    pub max_deviation_per_update: u64,
    pub max_total_deviation: u64,
    pub min_history_points: usize,
    pub min_bull_price: u64,
    pub min_bear_price: u64,
}

/// Enhanced price history for storing timestamps/prices/volumes
#[derive(Clone, AnchorSerialize, AnchorDeserialize)]
pub struct EnhancedPriceHistory {
    pub timestamps: VecDeque<i64>,
    pub prices: VecDeque<u64>,
    pub volumes: VecDeque<u64>,
    pub max_size: usize,
    pub min_valid_price: u64,
    pub last_update: i64,
    pub cumulative_volume: u64,
    pub twap_window: i64,
}

impl EnhancedPriceHistory {
    pub fn new(max_size: usize, min_price: u64, twap_window: i64) -> Self {
        Self {
            timestamps: VecDeque::with_capacity(max_size),
            prices: VecDeque::with_capacity(max_size),
            volumes: VecDeque::with_capacity(max_size),
            max_size,
            min_valid_price: min_price,
            last_update: 0,
            cumulative_volume: 0,
            twap_window,
        }
    }

    /// Push a new price+volume data point
    pub fn push(&mut self, ts: i64, price: u64, volume: u64) -> Result<()> {
        require!(price >= self.min_valid_price, ErrorCode::InvalidPrice);
        require!(ts >= self.last_update, ErrorCode::InvalidTimestamp);
        require!(volume > 0, ErrorCode::ZeroVolume);

        while let Some(&oldest_ts) = self.timestamps.front() {
            if ts.saturating_sub(oldest_ts) > self.twap_window {
                self.timestamps.pop_front();
                self.prices.pop_front();
                self.volumes.pop_front();
            } else {
                break;
            }
        }
        if self.timestamps.len() >= self.max_size {
            self.timestamps.pop_front();
            self.prices.pop_front();
            self.volumes.pop_front();
        }
        self.timestamps.push_back(ts);
        self.prices.push_back(price);
        self.volumes.push_back(volume);

        self.last_update = ts;
        self.cumulative_volume = self.cumulative_volume.saturating_add(volume);
        Ok(())
    }

    /// Calculate time-weighted average price
    pub fn calculate_twap(&self, window: i64) -> Result<u64> {
        let current_ts = Clock::get()?.unix_timestamp;
        let start_ts = current_ts.saturating_sub(window);
        let mut sum_price_volume = 0u128;
        let mut sum_volume = 0u64;

        for i in 0..self.timestamps.len() {
            if self.timestamps[i] >= start_ts {
                let price_vol = (self.prices[i] as u128)
                    .checked_mul(self.volumes[i] as u128)
                    .ok_or(ErrorCode::Overflow)?;
                sum_price_volume = sum_price_volume.checked_add(price_vol).ok_or(ErrorCode::Overflow)?;
                sum_volume = sum_volume.saturating_add(self.volumes[i]);
            }
        }
        if sum_volume == 0 {
            return Ok(self.prices.back().copied().unwrap_or(self.min_valid_price));
        }
        Ok((sum_price_volume / sum_volume as u128) as u64)
    }

    /// Calculate approximate volatility
    pub fn calculate_volatility(&self, window: i64) -> Result<u64> {
        let twap = self.calculate_twap(window)?;
        let mut sum_squared_dev = 0u128;
        let mut count = 0u64;

        let current_ts = Clock::get()?.unix_timestamp;
        let start_ts = current_ts.saturating_sub(window);

        for i in 0..self.timestamps.len() {
            if self.timestamps[i] >= start_ts {
                let deviation = if self.prices[i] > twap {
                    self.prices[i] - twap
                } else {
                    twap - self.prices[i]
                } as u128;
                let sq = deviation.checked_mul(deviation).ok_or(ErrorCode::Overflow)?;
                sum_squared_dev = sum_squared_dev.checked_add(sq).ok_or(ErrorCode::Overflow)?;
                count += 1;
            }
        }
        if count == 0 {
            return Ok(0);
        }
        let avg_dev = sum_squared_dev.checked_div(count as u128).ok_or(ErrorCode::Overflow)?;
        Ok(integer_sqrt(avg_dev))
    }
}

/// An aggregator that merges multiple feeds
#[derive(Clone, AnchorSerialize, AnchorDeserialize)]
pub struct EnhancedOracleAggregator {
    pub feeds: Vec<OracleFeed>,
    pub weights: Vec<u64>,
    pub confidence_thresholds: Vec<u64>,
    pub staleness_thresholds: Vec<i64>,
    pub deviation_thresholds: Vec<u64>,
    pub minimum_feeds: usize,
}

impl EnhancedOracleAggregator {
    /// Weighted average from all valid feeds.
    /// Now `feed_accounts` must contain the AccountInfos for every feed in `self.feeds`.
    pub fn get_price(&mut self, feed_accounts: &[AccountInfo]) -> Result<OraclePrice> {
        let mut valid_prices: Vec<(i64, u64, i64)> = Vec::new();
        let mut total_weight = 0u64;
        let clock_ts = Clock::get()?.unix_timestamp;

        for i in 0..self.feeds.len() {
            let feed_conf_threshold   = self.confidence_thresholds.get(i).copied().unwrap_or(0);
            let feed_staleness_thresh = self.staleness_thresholds.get(i).copied().unwrap_or(3600);
            let feed_dev_threshold    = self.deviation_thresholds.get(i).copied().unwrap_or(u64::MAX);
            let feed_weight           = self.weights.get(i).copied().unwrap_or(0);

            // Find the AccountInfo for this feed in the provided slice.
            let feed_account = feed_accounts
                .iter()
                .find(|acc| *acc.key == self.feeds[i].feed_pubkey)
                .ok_or(ErrorCode::MissingOracleAccount)?;
            // Verify that this account is owned by the Pyth program.
            require_keys_eq!(feed_account.owner, PYTH_PROGRAM_ID, ErrorCode::InvalidOwnerForPythAccount);

            let ephemeral_price = read_pyth_ephemeral(feed_account, &self.feeds[i].kind)?;
            if ephemeral_price.confidence > feed_conf_threshold {
                continue;
            }
            let age = clock_ts.saturating_sub(ephemeral_price.timestamp);
            if age > feed_staleness_thresh {
                continue;
            }
            if let FeedKind::Price = self.feeds[i].kind {
                if ephemeral_price.price < 0 {
                    continue;
                }
            }
            let prev_price = self.feeds[i].previous_price_i64;
            if prev_price != 0 {
                let diff = (ephemeral_price.price - prev_price).abs();
                if diff as u64 > feed_dev_threshold {
                    continue;
                }
            }
            valid_prices.push((ephemeral_price.price, feed_weight, ephemeral_price.timestamp));
        }

        require!(valid_prices.len() >= self.minimum_feeds, ErrorCode::InsufficientValidFeeds);

        for (_, weight, _) in &valid_prices {
            total_weight = total_weight.saturating_add(*weight);
        }
        require!(total_weight > 0, ErrorCode::InvalidPrice);

        let mut sum_price: i128 = 0;
        let mut max_ts = 0i64;
        for (p, w, t) in &valid_prices {
            let partial = (*p as i128).checked_mul(*w as i128).ok_or(ErrorCode::Overflow)?;
            sum_price = sum_price.checked_add(partial).ok_or(ErrorCode::Overflow)?;
            if *t > max_ts {
                max_ts = *t;
            }
        }
        let avg = sum_price.checked_div(total_weight as i128).ok_or(ErrorCode::Overflow)?;
        require!(avg >= -2_000_000_000 && avg <= 2_000_000_000, ErrorCode::InvalidPriceRange);

        // Update aggregator’s stored previous values.
        for i in 0..self.feeds.len() {
            let feed_weight = self.weights.get(i).copied().unwrap_or(0);
            if feed_weight == 0 {
                continue;
            }
            let feed_account = feed_accounts
                .iter()
                .find(|acc| *acc.key == self.feeds[i].feed_pubkey)
                .ok_or(ErrorCode::MissingOracleAccount)?;
            let ephemeral_price = read_pyth_ephemeral(feed_account, &self.feeds[i].kind)?;
            self.feeds[i].previous_price_i64 = ephemeral_price.price;
            self.feeds[i].previous_confidence = ephemeral_price.confidence;
        }

        Ok(OraclePrice {
            price: avg as i64,
            confidence: 0,
            timestamp: max_ts,
            status: PriceStatus::Trading,
        })
    }
}

/// NEW: Read Pyth price feed data from an AccountInfo.
/// This function uses the pyth_sdk_solana crate to deserialize the on‑chain data.
fn read_pyth_ephemeral(account_info: &AccountInfo, _kind: &FeedKind) -> Result<OraclePrice> {
    let data = account_info.data.borrow();
    let price_account = load_price_account(&data).map_err(|_| ErrorCode::InvalidOwnerForPythAccount)?;

    // Pyth SDK stores these fields in `price_account.agg`.
    let publish_time = price_account.timestamp;
    let expo = price_account.exponent;
    let raw_price = price_account.agg.price;
    let raw_conf = price_account.agg.confidence;

    // Adjust the raw price based on the exponent.
    let adjusted_price = if expo < 0 {
        let factor = 10i64.pow((-expo) as u32);
        raw_price.checked_mul(factor).ok_or(ErrorCode::Overflow)?
    } else if expo > 0 {
        let factor = 10i64.pow(expo as u32);
        raw_price.checked_div(factor).ok_or(ErrorCode::Overflow)?
    } else {
        raw_price
    };

    // Adjust the confidence similarly.
    let adjusted_conf = if expo < 0 {
        let factor = 10u64.pow((-expo) as u32);
        raw_conf.checked_mul(factor).ok_or(ErrorCode::Overflow)?
    } else if expo > 0 {
        let factor = 10u64.pow(expo as u32);
        raw_conf.checked_div(factor).ok_or(ErrorCode::Overflow)?
    } else {
        raw_conf
    };

    let clock_ts = Clock::get()?.unix_timestamp;
    let status = if clock_ts.saturating_sub(publish_time) > 60 {
        PriceStatus::Halted
    } else {
        PriceStatus::Trading
    };

    Ok(OraclePrice {
        price: adjusted_price,
        confidence: adjusted_conf,
        timestamp: publish_time,
        status,
    })
}

// ====================================================
// ADAPTIVE CIRCUIT BREAKER & GOVERNANCE STRUCTS (unchanged)
// ====================================================

#[derive(Clone, AnchorSerialize, AnchorDeserialize)]
pub struct AdaptiveCircuitBreaker {
    pub base_threshold: u64,
    pub volume_threshold: u64,
    pub market_state: MarketState,
}

#[derive(Clone, Copy, AnchorSerialize, AnchorDeserialize, PartialEq)]
pub enum MarketState {
    Normal,
    Recovering,
    Emergency,
}

impl AdaptiveCircuitBreaker {
    pub fn validate_price_update(&self, current_price: u64, new_price: u64, volume: u64) -> Result<()> {
        if current_price > 0 {
            let change_bp = (((new_price as i64) - (current_price as i64)) * 10000) / (current_price as i64);
            require!((change_bp.abs() as u64) <= self.base_threshold, ErrorCode::CircuitBreaker);
        }
        require!(volume >= self.volume_threshold, ErrorCode::InsufficientVolume);
        Ok(())
    }
}

#[derive(Clone, AnchorSerialize, AnchorDeserialize)]
pub struct AdvancedGovernance {
    pub authorities: OptimizedVecMap<Pubkey, Role>,
    pub proposals: OptimizedVecMap<u64, Proposal>,
    pub voting_config: VotingConfig,
    pub timelock_config: TimelockConfig,
    pub emergency_config: EmergencyConfig,
    pub next_proposal_id: u64,
    pub initial_governor: Pubkey,
}

impl AdvancedGovernance {
    pub fn is_authorized(&self, user: &Pubkey, required_role: Role) -> bool {
        fn role_priority(role: &Role) -> u8 {
            match role {
                Role::Admin => 3,
                Role::Operator => 2,
                Role::PricePusher => 1,
                Role::Treasury => 1,
            }
        }
        if let Some(current_role) = self.authorities.get(user) {
            role_priority(current_role) >= role_priority(&required_role)
        } else {
            false
        }
    }
}

#[derive(Clone, Copy, PartialEq, AnchorSerialize, AnchorDeserialize)]
pub enum Role {
    Admin,
    Operator,
    PricePusher,
    Treasury,
}

#[derive(Clone, AnchorSerialize, AnchorDeserialize)]
pub struct Proposal {
    pub id: u64,
    pub proposer: Pubkey,
    pub action: GovernanceAction,
    pub urgency: ActionUrgency,
    pub status: ProposalStatus,
    pub creation_time: i64,
    pub execution_time: Option<i64>,
    pub votes: Vec<Vote>,
    pub required_role: Role,
}

#[derive(Clone, AnchorSerialize, AnchorDeserialize)]
pub struct Vote {
    pub voter: Pubkey,
    pub support: bool,
    pub voting_power: u64,
    pub timestamp: i64,
}

#[derive(Clone, Copy, AnchorSerialize, AnchorDeserialize)]
pub enum ProposalStatus {
    Pending,
    Active,
    Succeeded,
    Queued,
    Executed,
    Cancelled,
    Failed,
    Expired,
}

#[derive(Clone, AnchorSerialize, AnchorDeserialize)]
pub enum GovernanceAction {
    AddPricePusher(Pubkey),
    RemovePricePusher(Pubkey),
    SetPricePusherStatus(Pubkey, bool),
    EmergencyPause,
    ResumeMarket,
}

impl GovernanceAction {
    pub fn required_role(&self) -> Role {
        match self {
            Self::AddPricePusher(_) => Role::Admin,
            Self::RemovePricePusher(_) => Role::Admin,
            Self::SetPricePusherStatus(_, _) => Role::Operator,
            Self::EmergencyPause => Role::Admin,
            Self::ResumeMarket => Role::Admin,
        }
    }
}

#[derive(Clone, Copy, AnchorSerialize, AnchorDeserialize)]
pub enum ActionUrgency {
    Normal,
    Critical,
    Emergency,
}

#[derive(Clone, AnchorSerialize, AnchorDeserialize)]
pub struct VotingConfig {
    pub quorum: u64,
    pub supermajority: u64,
    pub voting_period: i64,
    pub execution_delay: i64,
    pub min_voting_power: u64,
}

#[derive(Clone, AnchorSerialize, AnchorDeserialize)]
pub struct TimelockConfig {
    pub normal_delay: i64,
    pub critical_delay: i64,
    pub emergency_delay: i64,
}

#[derive(Clone, AnchorSerialize, AnchorDeserialize)]
pub struct EmergencyConfig {
    pub timelock: i64,
    pub required_approvals: u64,
    pub cool_down_period: i64,
}

// ====================================================
// PRICE / VALUE CALCULATIONS (unchanged)
// ====================================================

pub fn calculate_new_mint_price(
    base_price: u64,
    new_oracle_price: i64,
    prev_oracle_price: i64,
    quantity: u64,
    is_bull: bool,
    min_price: u64,
    config: &PriceConfig,
) -> Result<u64> {
    if prev_oracle_price == 0 {
        return Ok(base_price);
    }
    let relative_change_bp = ((new_oracle_price - prev_oracle_price) * 10000) / prev_oracle_price;
    let effective_change_bp = if is_bull {
        relative_change_bp
    } else {
        -relative_change_bp
    };
    let price_adjustment_bp = (effective_change_bp * config.weighting_spy as i64) / 10000;

    let new_price_oracle = (base_price as i128)
        .checked_mul((10000 + price_adjustment_bp) as i128)
        .ok_or(ErrorCode::Overflow)?
        .checked_div(10000)
        .ok_or(ErrorCode::Overflow)?;

    let increments = quantity / 10;
    let volume_multiplier_bp = 10000 + (increments * 100);
    let new_price_final = new_price_oracle
        .checked_mul(volume_multiplier_bp as i128)
        .ok_or(ErrorCode::Overflow)?
        .checked_div(10000)
        .ok_or(ErrorCode::Overflow)?;

    let final_price = new_price_final.max(min_price as i128);
    require!(final_price <= u64::MAX as i128, ErrorCode::Overflow);
    Ok(final_price as u64)
}

pub fn calculate_decayed_fraction_value(entry: &FractionEntry, current_day: u64) -> u64 {
    let days_passed = current_day.saturating_sub(entry.day);
    if days_passed >= 10 {
        0
    } else {
        entry.quantity.saturating_mul(100 - 10 * days_passed) / 100
    }
}

pub fn total_raw_fractions(pos: &UserPosition) -> u64 {
    pos.raw_total_fractions
}

pub fn distribute_daily_rewards(
    state: &mut MagaIndexState,
    day: u64,
    _is_bullish: bool
) -> Result<Option<Pubkey>> {
    for (_, pos) in state.user_positions.data.iter_mut() {
        pos.consolidate_if_needed(day);
    }
    let top_user = state.user_positions.data.iter()
        .max_by_key(|(_, pos)| pos.total_value(day))
        .map(|(k, _)| *k);
    Ok(top_user)
}

pub fn mint_daily_artwork(ctx: &Context<PerformUpkeep>, winner: Pubkey, day: u64) -> Result<()> {
    let winner_ata_account: Account<TokenAccount> =
        Account::try_from(&ctx.accounts.winner_ata).map_err(|_| ErrorCode::InvalidAssociatedTokenAccount)?;

    require_keys_eq!(winner_ata_account.owner, winner, ErrorCode::InvalidATAOwner);
    require_keys_eq!(winner_ata_account.mint, ctx.accounts.nft_mint.key(), ErrorCode::InvalidATAMint);

    let (mint_authority_pda, bump) = MagaIndexState::get_mint_authority_address(ctx.program_id);

    let signer_seeds: &[&[u8]] = &[
        b"MAGA_INDEX_REWARD_POOL",
        ctx.program_id.as_ref(),
        &[bump]
    ];
    let signer_seeds_array = [signer_seeds];

    let metadata_program_id = Pubkey::new_from_array(MPL_TOKEN_METADATA_ID.to_bytes());
    let nft_mint_key = ctx.accounts.nft_mint.key();
    let metadata_seeds = &[
        b"metadata",
        metadata_program_id.as_ref(),
        nft_mint_key.as_ref(),
    ];
    let (metadata_pda, _metadata_bump) = Pubkey::find_program_address(metadata_seeds, &metadata_program_id);

    let nft_mint_converted = Pubkey::new_from_array(nft_mint_key.to_bytes());
    let authority_converted = Pubkey::new_from_array(ctx.accounts.authority.key().to_bytes());
    let mint_authority_converted = Pubkey::new_from_array(mint_authority_pda.to_bytes());
    let metadata_pda_converted = Pubkey::new_from_array(metadata_pda.to_bytes());

    let ix_metadata = create_metadata_accounts_v3(
        metadata_program_id,
        metadata_pda_converted,
        nft_mint_converted,
        mint_authority_converted,
        authority_converted,
        mint_authority_converted,
        format!("Daily Artwork - Day {}", day),
        "DART".to_string(),
        "https://yourdomain.com/path/to/metadata.json".to_string(),
        Some(vec![Creator {
            address: mint_authority_converted,
            verified: true,
            share: 100,
        }]),
        500,
        true,
        true,
        None,
        None,
        None,
    );
    invoke_signed(
        &ix_metadata,
        &[
            ctx.accounts.token_metadata_program.to_account_info(),
            ctx.accounts.nft_mint.to_account_info(),
            ctx.accounts.mint_authority.to_account_info(),
            ctx.accounts.authority.to_account_info(),
            ctx.accounts.system_program.to_account_info(),
        ],
        &signer_seeds_array,
    )?;

    let cpi_accounts = MintTo {
        mint: ctx.accounts.nft_mint.to_account_info(),
        to: ctx.accounts.winner_ata.to_account_info(),
        authority: ctx.accounts.mint_authority.to_account_info(),
    };
    let cpi_ctx = CpiContext::new_with_signer(
        ctx.accounts.token_program.to_account_info(),
        cpi_accounts,
        &signer_seeds_array,
    );
    mint_to(cpi_ctx, 1)?;

    let glitch_mint_key = ctx.accounts.nft_mint.key();
    let master_edition_seeds = &[
        b"metadata",
        metadata_program_id.as_ref(),
        glitch_mint_key.as_ref(),
        b"edition",
    ];
    let (master_edition_pda, _edition_bump) =
        Pubkey::find_program_address(master_edition_seeds, &metadata_program_id);

    let master_edition_pda_converted = Pubkey::new_from_array(master_edition_pda.to_bytes());
    let token_program_converted = Pubkey::new_from_array(ctx.accounts.token_program.key().to_bytes());
    let ix_master = create_master_edition_v3(
        metadata_program_id,
        master_edition_pda_converted,
        nft_mint_converted,
        mint_authority_converted,
        authority_converted,
        metadata_pda_converted,
        token_program_converted,
        Some(0)
    );
    invoke_signed(
        &ix_master,
        &[
            ctx.accounts.token_metadata_program.to_account_info(),
            ctx.accounts.nft_mint.to_account_info(),
            ctx.accounts.mint_authority.to_account_info(),
            ctx.accounts.authority.to_account_info(),
            ctx.accounts.system_program.to_account_info(),
        ],
        &signer_seeds_array,
    )?;
    Ok(())
}

pub fn reset_all_user_positions(state: &mut MagaIndexState) {
    for (_, pos) in state.user_positions.data.iter_mut() {
        pos.reset();
    }
}

pub fn get_top_user_by_raw_fractions(state: &MagaIndexState) -> Option<Pubkey> {
    state.user_positions.data.iter()
        .max_by_key(|(_, pos)| pos.raw_total_fractions)
        .map(|(k, _)| *k)
}

pub fn update_yearly_aggregator(
    _state: &mut MagaIndexState,
    _day: u64
) -> Result<()> {
    Ok(())
}

fn user_bull_value(position: &UserPosition, day: u64) -> u64 {
    position.recent_entries
        .iter()
        .filter(|e| e.is_bull)
        .map(|e| calculate_decayed_fraction_value(e, day))
        .sum()
}
fn user_bear_value(position: &UserPosition, day: u64) -> u64 {
    position.recent_entries
        .iter()
        .filter(|e| !e.is_bull)
        .map(|e| calculate_decayed_fraction_value(e, day))
        .sum()
}
fn find_second_place(
    state: &MagaIndexState,
    day: u64,
    is_bullish: bool
) -> Option<Pubkey> {
    let mut user_values: Vec<(Pubkey, u64)> = state.user_positions.data.iter()
        .map(|(user_key, position)| {
            let side_val = if is_bullish {
                user_bull_value(position, day)
            } else {
                user_bear_value(position, day)
            };
            (*user_key, side_val)
        })
        .collect();

    user_values.sort_by_key(|&(_, v)| v);
    user_values.reverse();

    if user_values.len() < 2 {
        return None;
    }
    if user_values[1].1 == 0 {
        None
    } else {
        Some(user_values[1].0)
    }
}
fn find_winning_side_accounts(
    state: &MagaIndexState,
    day: u64,
    is_bullish: bool
) -> Vec<Pubkey> {
    let mut winners = Vec::new();
    for (user_pk, position) in &state.user_positions.data {
        let side_val = if is_bullish {
            user_bull_value(position, day)
        } else {
            user_bear_value(position, day)
        };
        if side_val > 0 {
            winners.push(*user_pk);
        }
    }
    winners
}
fn user_value(
    state: &MagaIndexState,
    user_pk: Pubkey,
    day: u64,
    is_bullish: bool
) -> u64 {
    match state.user_positions.get(&user_pk) {
        Some(pos) => {
            if is_bullish {
                user_bull_value(pos, day)
            } else {
                user_bear_value(pos, day)
            }
        }
        None => 0
    }
}
fn sum_side_value(
    state: &MagaIndexState,
    users: &[Pubkey],
    day: u64,
    is_bullish: bool
) -> u64 {
    users.iter().fold(0u64, |acc, &pk| {
        acc.saturating_add(user_value(state, pk, day, is_bullish))
    })
}

// ====================================================
// MAIN PROGRAM STATE
// ====================================================

#[derive(Clone, AnchorSerialize, AnchorDeserialize)]
pub struct VerifiableState {
    pub dummy: u8,
}

/// A specialized map for storing user data
#[derive(Default, Clone, AnchorSerialize, AnchorDeserialize)]
pub struct OptimizedVecMap<K, V> {
    pub data: HashMap<K, V>,
}
impl<K: Eq + std::hash::Hash + Ord + Copy, V> OptimizedVecMap<K, V> {
    pub fn new() -> Self {
        Self { data: HashMap::new() }
    }
    pub fn get(&self, key: &K) -> Option<&V> {
        self.data.get(key)
    }
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        self.data.get_mut(key)
    }
    pub fn insert(&mut self, key: K, value: V) -> Result<()> {
        self.data.insert(key, value);
        Ok(())
    }
    pub fn remove(&mut self, key: &K) {
        self.data.remove(key);
    }
    pub fn len(&self) -> usize {
        self.data.len()
    }
    pub fn prune_below(&mut self, min_key: &K) {
        self.data.retain(|&k, _| k >= *min_key);
    }
}

/// The main program state account
#[account]
pub struct MagaIndexState {
    pub authority: Pubkey,
    pub governance_authority: Pubkey,
    pub oracle_aggregator: EnhancedOracleAggregator,
    pub circuit_breaker: AdaptiveCircuitBreaker,
    pub gov: AdvancedGovernance,
    pub ver_state: VerifiableState,
    pub timelock_duration: i64,
    pub last_timelock_set: i64,
    pub mint_price_bull: u64,
    pub mint_price_bear: u64,
    pub is_market_open: bool,
    pub decay_rates: DecayRates,
    pub daily_decay_rate_bps: u64,
    pub fraction_base_value: u64,
    pub price_history: EnhancedPriceHistory,
    pub creator_wallet: Pubkey,
    pub last_oracle_price: i64,
    pub yearly_totals: OptimizedVecMap<u64, YearCheckpoint>,
    pub user_positions: OptimizedVecMap<Pubkey, UserPosition>,
    pub settled_days: OptimizedVecMap<u64, bool>,
    pub value_checkpoints: OptimizedVecMap<u64, u64>,
    pub price_conf: PriceConfig,
    pub price_pushers: OptimizedVecMap<Pubkey, PricePusherInfo>,
    pub batch_offset: u64,
    pub batch_in_progress_day: Option<u64>,
    pub reward_treasury: Pubkey,
}

impl MagaIndexState {
    pub const MAX_SIZE: usize = 8_192;

    pub fn get_mint_authority_address(program_id: &Pubkey) -> (Pubkey, u8) {
        Pubkey::find_program_address(
            &[
                b"MAGA_INDEX_REWARD_POOL",
                program_id.as_ref(),
            ],
            program_id
        )
    }

    pub fn validate_price_push(&self, pusher: &Pubkey, now: i64) -> Result<()> {
        let info = self.price_pushers.get(pusher).ok_or(ErrorCode::Unauthorized)?;
        require!(info.is_active, ErrorCode::InactivePusher);
        if now.saturating_sub(info.last_push) < 10 {
            return err!(ErrorCode::RateLimitExceeded);
        }
        Ok(())
    }
}

// ====================================================
// ACCOUNT CONTEXTS
// ====================================================

#[derive(Accounts)]
pub struct InitializeState<'info> {
    #[account(init, payer = authority, space = MagaIndexState::MAX_SIZE)]
    pub maga_index_state: Account<'info, MagaIndexState>,
    #[account(mut)]
    pub authority: Signer<'info>,
    #[account(mut)]
    pub reward_treasury: Signer<'info>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct MintFractionBatch<'info> {
    #[account(mut)]
    pub maga_index_state: Account<'info, MagaIndexState>,
    #[account(init, payer = payer, space = 8 + FractionAccount::MAX_SIZE)]
    pub fraction_account: Account<'info, FractionAccount>,
    #[account(mut)]
    pub payer: Signer<'info>,
    #[account(mut)]
    pub reward_treasury: Signer<'info>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct PerformUpkeepBatch<'info> {
    #[account(mut)]
    pub maga_index_state: Account<'info, MagaIndexState>,
    #[account(mut)]
    pub authority: Signer<'info>,
}

#[derive(Accounts)]
pub struct PerformUpkeep<'info> {
    #[account(mut)]
    pub maga_index_state: Account<'info, MagaIndexState>,
    #[account(mut)]
    pub authority: Signer<'info>,
    #[account(mut)]
    pub reward_treasury: Signer<'info>,
    #[account(mut)]
    pub nft_mint: Account<'info, Mint>,
    #[account(mut)]
    pub mint_authority: AccountInfo<'info>,
    /// CHECK: The winner's associated token account
    #[account(mut)]
    pub winner_ata: AccountInfo<'info>,
    #[account(mut)]
    pub edition_authority: Signer<'info>,
    #[account(address = MPL_TOKEN_METADATA_ID)]
    pub token_metadata_program: AccountInfo<'info>,
    #[account(address = token::ID)]
    pub token_program: Program<'info, Token>,
    #[account(address = system_program::ID)]
    pub system_program: Program<'info, System>,
    pub associated_token_program: Program<'info, AssociatedToken>,
    #[account(address = clock::ID)]
    pub clock_sysvar: Sysvar<'info, Clock>,
    #[account(address = rent::ID)]
    pub rent: Sysvar<'info, Rent>,
    #[account(address = instructions::ID)]
    pub instructions_sysvar: AccountInfo<'info>,
}

#[derive(Accounts)]
pub struct RegisterRetroactiveNFT<'info> {
    #[account(mut)]
    pub maga_index_state: Account<'info, MagaIndexState>,
    #[account(mut)]
    pub authority: Signer<'info>,
}

#[derive(Accounts)]
pub struct PerformYearlyGlitchAuction<'info> {
    #[account(mut)]
    pub maga_index_state: Account<'info, MagaIndexState>,
    #[account(mut)]
    pub payer: Signer<'info>,
    #[account(init, payer = payer, associated_token::mint = glitch_mint, associated_token::authority = top_user_account)]
    pub user_ata: Account<'info, TokenAccount>,
    pub glitch_mint: Account<'info, Mint>,
    /// CHECK: The top user
    pub top_user_account: AccountInfo<'info>,
    /// CHECK: The mint authority (PDA)
    #[account(mut)]
    pub mint_authority: AccountInfo<'info>,
    #[account(address = MPL_TOKEN_METADATA_ID)]
    pub token_metadata_program: AccountInfo<'info>,
    #[account(address = token::ID)]
    pub token_program: Program<'info, Token>,
    #[account(address = system_program::ID)]
    pub system_program: Program<'info, System>,
    pub associated_token_program: Program<'info, AssociatedToken>,
    #[account(address = clock::ID)]
    pub clock_sysvar: Sysvar<'info, Clock>,
    #[account(address = rent::ID)]
    pub rent: Sysvar<'info, Rent>,
    #[account(address = instructions::ID)]
    pub instructions_sysvar: AccountInfo<'info>,
}

/// In the adapted version, we remove the Switchboard feed account from the PushPrice context.
/// All required Pyth feed accounts must be passed in as remaining_accounts.
#[derive(Accounts)]
pub struct PushPrice<'info> {
    #[account(mut)]
    pub maga_index_state: Account<'info, MagaIndexState>,
    #[account(mut)]
    pub price_pusher: Signer<'info>,
    // All Pyth feed accounts are expected to be passed as remaining_accounts.
}

#[derive(Accounts)]
pub struct ManagePricePusher<'info> {
    #[account(mut)]
    pub maga_index_state: Account<'info, MagaIndexState>,
    #[account(mut)]
    pub authority: Signer<'info>,
}

// ====================================================
// MAIN PROGRAM MODULE
// ====================================================

#[program]
pub mod maga_index_advanced_plus_upgrade {
    use super::*;

    /// Initialize the full state
    pub fn initialize_full_state(
        ctx: Context<InitializeState>,
        aggregator_conf: EnhancedOracleAggregator,
        circuit_breaker: AdaptiveCircuitBreaker,
        gov: AdvancedGovernance,
        ver_state: VerifiableState,
        timelock_duration: i64,
        min_valid_price: u64,
        twap_window: i64,
        max_history_size: usize,
        decay_rates: DecayRates,
    ) -> Result<()> {
        msg!("Instruction: InitializeFullState");
        let state = &mut ctx.accounts.maga_index_state;
        state.authority = *ctx.accounts.authority.key;
        state.governance_authority = gov.initial_governor;
        state.oracle_aggregator = aggregator_conf;
        state.circuit_breaker = circuit_breaker;
        state.gov = gov;
        state.ver_state = ver_state;
        state.timelock_duration = timelock_duration;
        state.last_timelock_set = 0;
        state.mint_price_bull = 1_000_000_000;
        state.mint_price_bear = 1_000_000_000;
        state.is_market_open = true;
        state.decay_rates = decay_rates;
        state.daily_decay_rate_bps = 50;
        state.fraction_base_value = 1_0000;
        state.price_history = EnhancedPriceHistory::new(max_history_size, min_valid_price, twap_window);
        state.reward_treasury = ctx.accounts.reward_treasury.key();
        let clock = Clock::get()?;
        let bull_price = state.mint_price_bull;
        state.price_history.push(clock.unix_timestamp, bull_price, 1)?;
        state.last_oracle_price = state.mint_price_bull as i64;
        state.yearly_totals = OptimizedVecMap::new();
        state.user_positions = OptimizedVecMap::new();
        state.settled_days = OptimizedVecMap::new();
        state.value_checkpoints = OptimizedVecMap::new();
        state.price_conf = PriceConfig {
            weighting_spy: 1000,
            weighting_btc: 0,
            volume_weight: 0,
            max_deviation_per_update: 20000,
            max_total_deviation: 50000,
            min_history_points: 5,
            min_bull_price: min_valid_price,
            min_bear_price: min_valid_price,
        };
        state.price_pushers = OptimizedVecMap::new();
        state.batch_offset = 0;
        state.batch_in_progress_day = None;
        state.creator_wallet = ctx.accounts.reward_treasury.key();
        Ok(())
    }

    /// Mint fraction batch
    pub fn mint_fraction_batch(
        ctx: Context<MintFractionBatch>,
        day: u64,
        is_bull: bool,
        quantity: u64,
        max_price: u64,
        deadline: i64,
    ) -> Result<()> {
        msg!("Instruction: MintFractionBatch");
        let clock = Clock::get()?;
        let state = &mut ctx.accounts.maga_index_state;
        require!(state.is_market_open, ErrorCode::MarketClosed);
        require!(clock.unix_timestamp < TERM_END_TIMESTAMP, ErrorCode::TermEnded);
        require!(deadline >= clock.unix_timestamp + 120, ErrorCode::DeadlineTooSoon);
        require!(clock.unix_timestamp <= deadline, ErrorCode::Expired);
        require!(quantity > 0, ErrorCode::ZeroQuantity);

        let final_oracle_price = state.oracle_aggregator.get_price(ctx.remaining_accounts)?;
        let volume = quantity;
        let current_price = if is_bull { state.mint_price_bull } else { state.mint_price_bear };

        let calculated_price = calculate_new_mint_price(
            current_price,
            final_oracle_price.price,
            state.last_oracle_price,
            quantity,
            is_bull,
            if is_bull { state.price_conf.min_bull_price } else { state.price_conf.min_bear_price },
            &state.price_conf,
        )?;
        state.circuit_breaker.validate_price_update(current_price, calculated_price, volume)?;
        require!(calculated_price <= max_price, ErrorCode::PriceTooHigh);

        let required_total = calculated_price.checked_mul(quantity).ok_or(ErrorCode::Overflow)?;
        let ix = system_instruction::transfer(
            &ctx.accounts.payer.key(),
            &ctx.accounts.reward_treasury.key(),
            required_total,
        );
        invoke(
            &ix,
            &[
                ctx.accounts.payer.to_account_info(),
                ctx.accounts.reward_treasury.to_account_info(),
                ctx.accounts.system_program.to_account_info(),
            ],
        )?;

        ctx.accounts.fraction_account.owner = ctx.accounts.payer.key();
        let mut user_position = state
            .user_positions
            .get(&ctx.accounts.fraction_account.owner)
            .cloned()
            .unwrap_or_else(|| UserPosition {
                last_consolidated_day: day,
                consolidated_value: 0,
                raw_total_fractions: 0,
                recent_entries: VecDeque::with_capacity(30),
            });
        let new_entry = FractionEntry {
            day,
            quantity,
            initial_value: quantity,
            is_bull,
        };
        user_position.recent_entries.push_back(new_entry);
        user_position.raw_total_fractions = user_position
            .raw_total_fractions
            .checked_add(quantity)
            .ok_or(ErrorCode::Overflow)?;
        ctx.accounts.fraction_account.total_contributed = ctx
            .accounts
            .fraction_account
            .total_contributed
            .checked_add(required_total)
            .ok_or(ErrorCode::Overflow)?;
        ctx.accounts.fraction_account.last_update_timestamp = clock.unix_timestamp;
        ctx.accounts.fraction_account.is_active = true;
        state.user_positions.insert(ctx.accounts.fraction_account.owner, user_position)?;
        state.price_history.push(clock.unix_timestamp, calculated_price, volume)?;
        if is_bull {
            state.mint_price_bull = calculated_price;
        } else {
            state.mint_price_bear = calculated_price;
        }
        state.last_oracle_price = final_oracle_price.price;
        Ok(())
    }

    /// Perform partial batch for day “day” over at most max_users
    pub fn perform_upkeep_batch(ctx: Context<PerformUpkeepBatch>, day: u64, max_users: u64) -> Result<()> {
        msg!("Instruction: PerformUpkeepBatch");
        let state = &mut ctx.accounts.maga_index_state;
        require!(
            state.gov.is_authorized(&ctx.accounts.authority.key(), Role::Operator),
            ErrorCode::Unauthorized
        );
        if state.batch_in_progress_day.is_none() {
            state.batch_in_progress_day = Some(day);
            state.batch_offset = 0;
        } else {
            let in_prog_day = state.batch_in_progress_day.unwrap();
            require_eq!(in_prog_day, day, ErrorCode::InvalidDay);
        }
        let mut user_keys: Vec<Pubkey> = state.user_positions.data.keys().copied().collect();
        user_keys.sort();
        let total_users = user_keys.len() as u64;
        if state.batch_offset >= total_users {
            return Ok(());
        }
        let end_index = state.batch_offset.saturating_add(max_users).min(total_users);
        for i in state.batch_offset..end_index {
            let user_pk = user_keys[i as usize];
            if let Some(pos) = state.user_positions.get_mut(&user_pk) {
                pos.consolidate_if_needed(day);
            }
        }
        state.batch_offset = end_index;
        Ok(())
    }

    /// Final step to distribute lamports/mint NFT once batch is done.
    pub fn perform_upkeep(ctx: Context<PerformUpkeep>, day: u64) -> Result<()> {
        msg!("Instruction: PerformUpkeep");
        let state = &mut ctx.accounts.maga_index_state;
        require!(
            state.gov.is_authorized(&ctx.accounts.authority.key(), Role::Operator),
            ErrorCode::Unauthorized
        );
        let already_settled = state.settled_days.get(&day).copied().unwrap_or(false);
        require!(!already_settled, ErrorCode::DayAlreadySettled);
        let current_ts = Clock::get()?.unix_timestamp;
        let expected_day = current_ts / 86400;
        require_eq!(day as i64, expected_day, ErrorCode::InvalidDay);
        require!(state.batch_in_progress_day == Some(day), ErrorCode::InvalidDay);
        let total_users = state.user_positions.data.len() as u64;
        require!(state.batch_offset >= total_users, ErrorCode::DayNotYetSettledInBatch);

        state.batch_in_progress_day = None;
        state.batch_offset = 0;
        let (maybe_winner, maybe_second, side_accounts, is_bullish_day_local) = {
            let final_oracle_price = state.oracle_aggregator.get_price(ctx.remaining_accounts)?;
            let is_bullish_day_local = final_oracle_price.price > 0;
            let maybe_winner_local = distribute_daily_rewards(state, day, is_bullish_day_local)?;
            let maybe_second_local = find_second_place(state, day, is_bullish_day_local);
            let side_local = find_winning_side_accounts(state, day, is_bullish_day_local);
            state.settled_days.insert(day, true)?;
            update_yearly_aggregator(state, day)?;
            state.mint_price_bull = state.price_conf.min_bull_price;
            state.mint_price_bear = state.price_conf.min_bear_price;
            let oldest_day_to_keep = day.saturating_sub(365);
            state.value_checkpoints.prune_below(&oldest_day_to_keep);
            state.settled_days.prune_below(&oldest_day_to_keep);
            (maybe_winner_local, maybe_second_local, side_local, is_bullish_day_local)
        };

        let reward_treasury_balance = ctx.accounts.reward_treasury.lamports();
        let rent_exempt_reserve = 0;
        let distributable = reward_treasury_balance.saturating_sub(rent_exempt_reserve);

        if distributable > 0 {
            let creator_cut = distributable
                .checked_mul(70).ok_or(ErrorCode::Overflow)?
                .checked_div(100).ok_or(ErrorCode::Overflow)?;
            let winner_cut  = distributable
                .checked_mul(15).ok_or(ErrorCode::Overflow)?
                .checked_div(100).ok_or(ErrorCode::Overflow)?;
            let second_cut  = distributable
                .checked_mul(10).ok_or(ErrorCode::Overflow)?
                .checked_div(100).ok_or(ErrorCode::Overflow)?;
            let side_cut    = distributable
                .checked_mul(5).ok_or(ErrorCode::Overflow)?
                .checked_div(100).ok_or(ErrorCode::Overflow)?;
            {
                let ix = system_instruction::transfer(
                    &ctx.accounts.reward_treasury.key(),
                    &state.creator_wallet,
                    creator_cut,
                );
                invoke(
                    &ix,
                    &[
                        ctx.accounts.reward_treasury.to_account_info(),
                        ctx.accounts.system_program.to_account_info(),
                    ],
                )?;
            }
            if let Some(winner_pk) = maybe_winner {
                let ix = system_instruction::transfer(
                    &ctx.accounts.reward_treasury.key(),
                    &winner_pk,
                    winner_cut,
                );
                invoke(
                    &ix,
                    &[
                        ctx.accounts.reward_treasury.to_account_info(),
                        ctx.accounts.system_program.to_account_info(),
                    ],
                )?;
            }
            if let Some(second_pk) = maybe_second {
                let ix = system_instruction::transfer(
                    &ctx.accounts.reward_treasury.key(),
                    &second_pk,
                    second_cut,
                );
                invoke(
                    &ix,
                    &[
                        ctx.accounts.reward_treasury.to_account_info(),
                        ctx.accounts.system_program.to_account_info(),
                    ],
                )?;
            }
            if side_cut > 0 && !side_accounts.is_empty() {
                let side_total = {
                    let st = &ctx.accounts.maga_index_state;
                    sum_side_value(st, &side_accounts, day, is_bullish_day_local)
                };
                if side_total > 0 {
                    for user_pk in &side_accounts {
                        let user_val = {
                            let st = &ctx.accounts.maga_index_state;
                            user_value(st, *user_pk, day, is_bullish_day_local)
                        };
                        let share = side_cut.saturating_mul(user_val) / side_total;
                        if share > 0 {
                            let ix = system_instruction::transfer(
                                &ctx.accounts.reward_treasury.key(),
                                user_pk,
                                share,
                            );
                            invoke(
                                &ix,
                                &[
                                    ctx.accounts.reward_treasury.to_account_info(),
                                    ctx.accounts.system_program.to_account_info(),
                                ],
                            )?;
                        }
                    }
                }
            }
        }

        if let Some(winner_pubkey) = maybe_winner {
            mint_daily_artwork(&ctx, winner_pubkey, day)?;
        }
        Ok(())
    }

    /// Perform a yearly glitch auction
    pub fn perform_yearly_glitch_auction(
        ctx: Context<PerformYearlyGlitchAuction>,
        _top_user: Pubkey,
        year: u64,
        metadata_uri_input: String,
    ) -> Result<()> {
        msg!("Instruction: PerformYearlyGlitchAuction");
        require!(matches!(year, 2026 | 2027 | 2028 | 2029), ErrorCode::InvalidYearForAuction);
        let ts = Clock::get()?.unix_timestamp;
        let start_date = match year {
            2026 => GLITCH_AUCTION_DATE_2026,
            2027 => GLITCH_AUCTION_DATE_2027,
            2028 => GLITCH_AUCTION_DATE_2028,
            2029 => GLITCH_AUCTION_DATE_2029,
            _ => return err!(ErrorCode::InvalidYearForAuction),
        };
        require!(ts >= start_date && ts < start_date + 86400, ErrorCode::AuctionNotActive);
        let _top_user = get_top_user_by_raw_fractions(&ctx.accounts.maga_index_state)
            .ok_or(ErrorCode::NoUserFound)?;
        let (mint_authority_pda, bump) = MagaIndexState::get_mint_authority_address(&ctx.program_id);
        let signer_seeds: &[&[u8]] = &[
            b"MAGA_INDEX_REWARD_POOL",
            ctx.program_id.as_ref(),
            &[bump],
        ];
        let signer_seeds_array = [signer_seeds];
        let metadata_program_id = Pubkey::new_from_array(MPL_TOKEN_METADATA_ID.to_bytes());
        let glitch_mint_key = ctx.accounts.glitch_mint.key();
        let metadata_seeds = &[
            b"metadata",
            metadata_program_id.as_ref(),
            glitch_mint_key.as_ref(),
        ];
        let (metadata_pda, _metadata_bump) =
            Pubkey::find_program_address(metadata_seeds, &metadata_program_id);
        let master_edition_seeds = &[
            b"metadata",
            metadata_program_id.as_ref(),
            glitch_mint_key.as_ref(),
            b"edition",
        ];
        let (master_edition_pda, _edition_bump) =
            Pubkey::find_program_address(master_edition_seeds, &metadata_program_id);
        let name = format!("Glitch NFT - Year {}", year);
        let symbol = "GLITCH".to_string();
        let nft_mint_converted = Pubkey::new_from_array(glitch_mint_key.to_bytes());
        let payer_converted = Pubkey::new_from_array(ctx.accounts.payer.key().to_bytes());
        let mint_authority_converted = Pubkey::new_from_array(mint_authority_pda.to_bytes());
        let metadata_pda_converted = Pubkey::new_from_array(metadata_pda.to_bytes());
        let ix_metadata = create_metadata_accounts_v3(
            metadata_program_id,
            metadata_pda_converted,
            nft_mint_converted,
            mint_authority_converted,
            payer_converted,
            mint_authority_converted,
            name,
            symbol,
            metadata_uri_input,
            Some(vec![Creator {
                address: mint_authority_converted,
                verified: true,
                share: 100,
            }]),
            500,
            true,
            true,
            None,
            None,
            None,
        );
        invoke_signed(
            &ix_metadata,
            &[
                ctx.accounts.token_metadata_program.to_account_info(),
                ctx.accounts.glitch_mint.to_account_info(),
                ctx.accounts.mint_authority.to_account_info(),
                ctx.accounts.payer.to_account_info(),
                ctx.accounts.system_program.to_account_info(),
            ],
            &signer_seeds_array,
        )?;
        let cpi_accounts = MintTo {
            mint: ctx.accounts.glitch_mint.to_account_info(),
            to: ctx.accounts.user_ata.to_account_info(),
            authority: ctx.accounts.mint_authority.to_account_info(),
        };
        let cpi_ctx = CpiContext::new_with_signer(
            ctx.accounts.token_program.to_account_info(),
            cpi_accounts,
            &signer_seeds_array,
        );
        mint_to(cpi_ctx, 1)?;
        let master_edition_pda_converted = Pubkey::new_from_array(master_edition_pda.to_bytes());
        let token_program_converted = Pubkey::new_from_array(ctx.accounts.token_program.key().to_bytes());
        let ix_master = create_master_edition_v3(
            metadata_program_id,
            master_edition_pda_converted,
            nft_mint_converted,
            mint_authority_converted,
            payer_converted,
            metadata_pda_converted,
            token_program_converted,
            Some(0)
        );
        invoke_signed(
            &ix_master,
            &[
                ctx.accounts.token_metadata_program.to_account_info(),
                ctx.accounts.glitch_mint.to_account_info(),
                ctx.accounts.mint_authority.to_account_info(),
                ctx.accounts.payer.to_account_info(),
                ctx.accounts.system_program.to_account_info(),
            ],
            &signer_seeds_array,
        )?;
        reset_all_user_positions(&mut ctx.accounts.maga_index_state);
        Ok(())
    }

    /// Register a retroactive NFT
    pub fn register_retroactive_nft(
        ctx: Context<RegisterRetroactiveNFT>,
        _day: u64,
        _mint: Pubkey,
        _metadata_uri: String,
    ) -> Result<()> {
        msg!("Instruction: RegisterRetroactiveNFT");
        let state = &mut ctx.accounts.maga_index_state;
        require!(state.gov.is_authorized(&ctx.accounts.authority.key(), Role::Admin), ErrorCode::Unauthorized);
        Ok(())
    }

    /// Push a price update from Pyth.
    /// In this adapted version, all required Pyth feed accounts must be passed in as remaining_accounts.
    pub fn push_price_update(ctx: Context<PushPrice>) -> Result<()> {
        msg!("Instruction: PushPriceUpdate");
        let state = &mut ctx.accounts.maga_index_state;
        require!(state.is_market_open, ErrorCode::MarketClosed);
        require!(state.gov.is_authorized(&ctx.accounts.price_pusher.key(), Role::PricePusher), ErrorCode::Unauthorized);
        // Ensure each remaining account is owned by the Pyth program.
        for acc in ctx.remaining_accounts.iter() {
            require_keys_eq!(acc.owner, PYTH_PROGRAM_ID, ErrorCode::InvalidOwnerForPythAccount);
        }
        let now = Clock::get()?.unix_timestamp;
        state.validate_price_push(&ctx.accounts.price_pusher.key(), now)?;
        // Pass all remaining accounts (the Pyth feed accounts) into the aggregator.
        let final_oracle_price = state.oracle_aggregator.get_price(ctx.remaining_accounts)?;
        let volume = 1;
        let prev_price = if state.last_oracle_price == 0 {
            final_oracle_price.price
        } else {
            state.last_oracle_price
        };
        let new_bull_price = calculate_new_mint_price(
            state.mint_price_bull,
            final_oracle_price.price,
            prev_price,
            volume,
            true,
            state.price_conf.min_bull_price,
            &state.price_conf,
        )?;
        let new_bear_price = calculate_new_mint_price(
            state.mint_price_bear,
            final_oracle_price.price,
            prev_price,
            volume,
            false,
            state.price_conf.min_bear_price,
            &state.price_conf,
        )?;
        state.circuit_breaker.validate_price_update(state.mint_price_bull, new_bull_price, volume)?;
        state.circuit_breaker.validate_price_update(state.mint_price_bear, new_bear_price, volume)?;
        state.mint_price_bull = new_bull_price;
        state.mint_price_bear = new_bear_price;
        state.last_oracle_price = final_oracle_price.price;

        let mut pusher_info = state
            .price_pushers
            .get(&ctx.accounts.price_pusher.key())
            .cloned()
            .unwrap_or_else(|| PricePusherInfo {
                last_push: 0,
                push_count: 0,
                is_active: true,
            });
        pusher_info.last_push = now;
        pusher_info.push_count = pusher_info.push_count.saturating_add(1);
        state.price_pushers.insert(ctx.accounts.price_pusher.key(), pusher_info)?;
        state.price_history.push(now, new_bull_price, volume)?;
        Ok(())
    }

    /// Manage Price Pusher - add
    pub fn add_price_pusher(ctx: Context<ManagePricePusher>, pusher: Pubkey) -> Result<()> {
        msg!("Instruction: AddPricePusher");
        let state = &mut ctx.accounts.maga_index_state;
        require!(state.gov.is_authorized(&ctx.accounts.authority.key(), Role::Admin), ErrorCode::Unauthorized);
        require!(state.price_pushers.get(&pusher).is_none(), ErrorCode::PusherAlreadyExists);
        let info = PricePusherInfo { last_push: 0, push_count: 0, is_active: true };
        state.price_pushers.insert(pusher, info)?;
        state.gov.authorities.insert(pusher, Role::PricePusher)?;
        Ok(())
    }

    /// Manage Price Pusher - remove
    pub fn remove_price_pusher(ctx: Context<ManagePricePusher>, pusher: Pubkey) -> Result<()> {
        msg!("Instruction: RemovePricePusher");
        let state = &mut ctx.accounts.maga_index_state;
        require!(state.gov.is_authorized(&ctx.accounts.authority.key(), Role::Admin), ErrorCode::Unauthorized);
        state.price_pushers.remove(&pusher);
        state.gov.authorities.remove(&pusher);
        Ok(())
    }

    /// Manage Price Pusher - update status
    pub fn update_pusher_status(ctx: Context<ManagePricePusher>, pusher: Pubkey, is_active: bool) -> Result<()> {
        msg!("Instruction: UpdatePusherStatus");
        let state = &mut ctx.accounts.maga_index_state;
        require!(state.gov.is_authorized(&ctx.accounts.authority.key(), Role::Operator), ErrorCode::Unauthorized);
        let mut info = state.price_pushers.get(&pusher).cloned().ok_or(ErrorCode::PusherNotFound)?;
        info.is_active = is_active;
        state.price_pushers.insert(pusher, info)?;
        Ok(())
    }
}

// ====================================================
// HELPER FUNCTION OUTSIDE THE PROGRAM MODULE
// ====================================================

pub fn transfer_from_treasury<'info>(
    treasury_account: &AccountInfo<'info>,
    system_program: &AccountInfo<'info>,
    recipient: Pubkey,
    lamports: u64,
) -> Result<()> {
    let ix = system_instruction::transfer(treasury_account.key, &recipient, lamports);
    invoke(
        &ix,
        &[
            treasury_account.clone(),
            system_program.clone(),
        ],
    )?;
    Ok(())
}

// ====================================================
// ERROR CODES
// ====================================================

#[error_code]
pub enum ErrorCode {
    #[msg("Unauthorized access")]
    Unauthorized,
    #[msg("Arithmetic overflow")]
    Overflow,
    #[msg("Invalid timestamp order")]
    InvalidTimeOrder,
    #[msg("Market is closed")]
    MarketClosed,
    #[msg("Market is in emergency state")]
    MarketEmergency,
    #[msg("Day already settled")]
    DayAlreadySettled,
    #[msg("Day not yet settled")]
    DayNotSettled,
    #[msg("Circuit breaker triggered")]
    CircuitBreaker,
    #[msg("Excessive price change")]
    ExcessivePriceChange,
    #[msg("Invalid price")]
    InvalidPrice,
    #[msg("Invalid price range")]
    InvalidPriceRange,
    #[msg("Price is too high")]
    PriceTooHigh,
    #[msg("Stale price")]
    StalePrice,
    #[msg("Invalid confidence range")]
    InvalidConfidence,
    #[msg("Invalid price status")]
    InvalidPriceStatus,
    #[msg("Insufficient valid feeds")]
    InsufficientValidFeeds,
    #[msg("Zero volume")]
    ZeroVolume,
    #[msg("Insufficient volume")]
    InsufficientVolume,
    #[msg("Insufficient recovery volume")]
    InsufficientRecoveryVolume,
    #[msg("Zero quantity")]
    ZeroQuantity,
    #[msg("Excessive volatility")]
    ExcessiveVolatility,
    #[msg("Timelock still active")]
    TimelockActive,
    #[msg("Deadline too soon")]
    DeadlineTooSoon,
    #[msg("Transaction expired")]
    Expired,
    #[msg("Rate limit exceeded")]
    RateLimitExceeded,
    #[msg("Inactive price pusher")]
    InactivePusher,
    #[msg("Unauthorized price pusher")]
    UnauthorizedPusher,
    #[msg("Pusher already exists")]
    PusherAlreadyExists,
    #[msg("Pusher not found")]
    PusherNotFound,
    #[msg("Proposal not found")]
    ProposalNotFound,
    #[msg("Invalid proposal status")]
    InvalidProposalStatus,
    #[msg("Voting period ended")]
    VotingPeriodEnded,
    #[msg("Already voted")]
    AlreadyVoted,
    #[msg("Insufficient voting power")]
    InsufficientVotingPower,
    #[msg("Quorum not reached")]
    QuorumNotReached,
    #[msg("Insufficient votes")]
    InsufficientVotes,
    #[msg("Map is full")]
    MapFull,
    #[msg("Invalid timestamp")]
    InvalidTimestamp,
    #[msg("Merkle path invalid")]
    InvalidMerklePath,
    #[msg("Invalid state hash")]
    InvalidStateHash,
    #[msg("Dependency not met")]
    DependencyNotMet,
    #[msg("Exceeded funds cap")]
    ExceededFundsCap,
    #[msg("Term has ended")]
    TermEnded,
    #[msg("Invalid year for auction")]
    InvalidYearForAuction,
    #[msg("Auction not active")]
    AuctionNotActive,
    #[msg("Reward pool account does not match expected PDA")]
    InvalidRewardPool,
    #[msg("Provided account is not a valid associated token account")]
    InvalidAssociatedTokenAccount,
    #[msg("ATA is not owned by the correct owner")]
    InvalidATAOwner,
    #[msg("ATA mint does not match the expected NFT mint")]
    InvalidATAMint,
    #[msg("Pyth feed account is not owned by the Pyth program")]
    InvalidOwnerForPythAccount,
    #[msg("Missing required oracle account")]
    MissingOracleAccount,
    #[msg("No user found")]
    NoUserFound,
    #[msg("Day is not yet fully processed by batch calls")]
    DayNotYetSettledInBatch,
    #[msg("Invalid day")]
    InvalidDay,
}

/// For testing or reference
pub mod test_exports {
    pub use crate::*;
    pub use anchor_lang::solana_program::program::{invoke, invoke_signed};
    pub use anchor_lang::solana_program::system_instruction;
    pub use mpl_token_metadata::instruction::{create_metadata_accounts_v3, create_master_edition_v3};
    pub use mpl_token_metadata::state::Creator;
    pub use std::collections::VecDeque;
}
