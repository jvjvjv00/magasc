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
use std::cmp::{min, max};

use pyth_sdk_solana::state::load_price_account; // Using the new Pyth SDK

/// The program ID for this contract - ensure this matches your intended ID
declare_id!("GL2qB66X6cNEBB7y5U7C4R6kEtG3brQs3MVZ6GaaS9Cy");

// ====================================================
// CONSTANTS & PLACEHOLDERS
// ====================================================

/// Terminal timestamp for contract functionality
pub const TERM_END_TIMESTAMP: i64 = 1_900_000_000;

/// Annual glitch auction dates
pub const GLITCH_AUCTION_DATE_2026: i64 = 1_750_000_000;
pub const GLITCH_AUCTION_DATE_2027: i64 = 1_780_000_000;
pub const GLITCH_AUCTION_DATE_2028: i64 = 1_810_000_000;
pub const GLITCH_AUCTION_DATE_2029: i64 = 1_840_000_000;

/// Maximum staleness threshold for oracle data in seconds
pub const MAX_ORACLE_STALENESS: i64 = 300; // 5 minutes 

/// Maximum size for user position maps to prevent resource exhaustion
pub const MAX_USER_POSITIONS: usize = 10000;

/// The Metaplex Token Metadata program ID
pub const MPL_TOKEN_METADATA_ID: Pubkey = mpl_token_metadata::ID;

/// For reading Pyth data (the program ID for Pyth on Solana)
pub const PYTH_PROGRAM_ID: Pubkey = Pubkey::new_from_array([
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2,
]);

// Event definitions for key state changes
#[event]
pub struct PriceUpdated {
    pub bull_price: u64,
    pub bear_price: u64,
    pub oracle_price: i64,
    pub timestamp: i64,
}

#[event]
pub struct FractionMinted {
    pub user: Pubkey,
    pub day: u64,
    pub is_bull: bool,
    pub quantity: u64,
    pub price: u64,
    pub timestamp: i64,
}

#[event]
pub struct RewardDistributed {
    pub day: u64,
    pub winner: Option<Pubkey>,
    pub runner_up: Option<Pubkey>,
    pub total_amount: u64,
    pub timestamp: i64,
}

#[event]
pub struct NFTMinted {
    pub recipient: Pubkey,
    pub mint: Pubkey,
    pub day: u64,
    pub timestamp: i64,
}

// ====================================================
// HELPER FUNCTIONS & UTILITIES
// ====================================================

/// Safely computes the integer square root of a u128
/// This implementation uses binary search to avoid overflow
fn integer_sqrt(value: u128) -> u64 {
    if value == 0 {
        return 0;
    }
    
    if value <= u64::MAX as u128 {
        // Fast path for small values
        return (value as f64).sqrt() as u64;
    }
    
    // Binary search for large values
    let mut lo: u128 = 0;
    let mut hi: u128 = min(value, u64::MAX as u128);
    
    while lo <= hi {
        let mid = lo + (hi - lo) / 2;
        let mid_squared = mid.saturating_mul(mid);
        
        if mid_squared == value {
            return mid as u64;
    }
    
    pub fn check_timelock(&self, action: &GovernanceAction, now: i64, last_timelock: i64) -> Result<()> {
        let delay = match action.urgency() {
            ActionUrgency::Normal => self.timelock_config.normal_delay,
            ActionUrgency::Critical => self.timelock_config.critical_delay,
            ActionUrgency::Emergency => self.timelock_config.emergency_delay,
        };
        
        if now < last_timelock + delay {
            return err!(ErrorCode::TimelockActive);
        }
        
        Ok(())
    } else if mid_squared < value {
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    
    // Return the largest value whose square doesn't exceed the input
    min(hi as u64, u64::MAX)
}

/// Computes the absolute difference between two u64 values
/// Returns the result as a u64, safely handling potential underflow
fn abs_diff(a: u64, b: u64) -> u64 {
    if a > b { a - b } else { b - a }
}

/// Validates an exponent is within safe range to prevent overflow
fn validate_exponent(expo: i32) -> Result<()> {
    require!(expo >= -18 && expo <= 18, ErrorCode::InvalidExponent);
    Ok(())
}

/// Safe mathematical operations with bounds checking
mod safe_math {
    use super::*;
    
    pub fn mul_div(a: u64, b: u64, denominator: u64) -> Result<u64> {
        if denominator == 0 {
            return err!(ErrorCode::DivisionByZero);
        }
        
        let a_u128 = a as u128;
        let b_u128 = b as u128;
        let denominator_u128 = denominator as u128;
        
        let product = a_u128.checked_mul(b_u128)
            .ok_or(ErrorCode::Overflow)?;
            
        let result = product.checked_div(denominator_u128)
            .ok_or(ErrorCode::DivisionByZero)?;
            
        if result > u64::MAX as u128 {
            return err!(ErrorCode::Overflow);
        }
        
        Ok(result as u64)
    }
    
    pub fn checked_percentage(value: u64, percentage_bps: i64) -> Result<u64> {
        require!(percentage_bps >= -10000 && percentage_bps <= 10000, 
                ErrorCode::InvalidPercentage);
        
        let value_i128 = value as i128;
        let percentage_i128 = percentage_bps as i128;
        
        let result = value_i128
            .checked_mul(10000 + percentage_i128)
            .ok_or(ErrorCode::Overflow)?
            .checked_div(10000)
            .ok_or(ErrorCode::DivisionByZero)?;
            
        require!(result >= 0 && result <= u64::MAX as i128, ErrorCode::Overflow);
        
        Ok(result as u64)
    }
}

// ====================================================
// EPHEMERAL STRUCTS (NO ANCHOR DERIVES)
// ====================================================

/// Enum representing the current trading status of a price feed
#[derive(Clone, Debug, PartialEq)]
pub enum PriceStatus {
    Trading,
    Halted,
    Unknown,
}

/// A struct for returning an oracle price with metadata
#[derive(Clone, Debug)]
pub struct OraclePrice {
    pub price: i64,
    pub confidence: u64,
    pub timestamp: i64,
    pub status: PriceStatus,
}

/// Distinguish between different types of oracle feeds
#[derive(Clone, AnchorSerialize, AnchorDeserialize, PartialEq)]
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
    pub bump: u8, // Added bump seed for verification
}

impl FractionAccount {
    pub const MAX_SIZE: usize = 8 + 32 + 8 + 8 + 1 + 1;
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

/// Info for a "price pusher" (someone allowed to push oracle updates)
#[derive(Clone, AnchorSerialize, AnchorDeserialize)]
pub struct PricePusherInfo {
    pub last_push: i64,
    pub push_count: u64,
    pub is_active: bool,
    pub min_interval: i64, // Minimum seconds between pushes
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
    pub staleness_threshold: i64, // Maximum age of price data in seconds
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

        // Remove timestamps older than the window
        while let Some(&oldest_ts) = self.timestamps.front() {
            if ts.saturating_sub(oldest_ts) > self.twap_window {
                self.timestamps.pop_front();
                self.prices.pop_front();
                self.volumes.pop_front();
            } else {
                break;
            }
        }
        
        // Enforce max size by removing oldest entries if needed
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
        // Input validation
        require!(window > 0, ErrorCode::InvalidParameter);
        require!(!self.timestamps.is_empty(), ErrorCode::InsufficientData);
        
        let current_ts = Clock::get()?.unix_timestamp;
        let start_ts = current_ts.saturating_sub(window);
        let mut sum_price_volume: u128 = 0;
        let mut sum_volume: u64 = 0;

        for i in 0..self.timestamps.len() {
            if self.timestamps[i] >= start_ts {
                let price = self.prices[i] as u128;
                let volume = self.volumes[i] as u128;
                
                let price_vol = price.checked_mul(volume)
                    .ok_or(ErrorCode::Overflow)?;
                    
                sum_price_volume = sum_price_volume.checked_add(price_vol)
                    .ok_or(ErrorCode::Overflow)?;
                    
                sum_volume = sum_volume.saturating_add(self.volumes[i]);
            }
        }
        
        if sum_volume == 0 {
            return Ok(self.prices.back().copied().unwrap_or(self.min_valid_price));
        }
        
        let result = sum_price_volume.checked_div(sum_volume as u128)
            .ok_or(ErrorCode::DivisionByZero)?;
            
        if result > u64::MAX as u128 {
            return err!(ErrorCode::Overflow);
        }
        
        Ok(result as u64)
    }

    /// Calculate approximate volatility
    pub fn calculate_volatility(&self, window: i64) -> Result<u64> {
        // Input validation
        require!(window > 0, ErrorCode::InvalidParameter);
        require!(self.timestamps.len() >= 2, ErrorCode::InsufficientData);
        
        let twap = self.calculate_twap(window)?;
        let mut sum_squared_dev: u128 = 0;
        let mut count: u64 = 0;

        let current_ts = Clock::get()?.unix_timestamp;
        let start_ts = current_ts.saturating_sub(window);

        for i in 0..self.timestamps.len() {
            if self.timestamps[i] >= start_ts {
                let deviation = if self.prices[i] > twap {
                    self.prices[i] - twap
                } else {
                    twap - self.prices[i]
                } as u128;
                
                // Safe square calculation
                if deviation > (u64::MAX as u128).sqrt() {
                    return err!(ErrorCode::Overflow);
                }
                
                let sq = deviation.checked_mul(deviation)
                    .ok_or(ErrorCode::Overflow)?;
                    
                sum_squared_dev = sum_squared_dev.checked_add(sq)
                    .ok_or(ErrorCode::Overflow)?;
                    
                count += 1;
            }
        }
        
        if count == 0 {
            return Ok(0);
        }
        
        let avg_dev = sum_squared_dev.checked_div(count as u128)
            .ok_or(ErrorCode::DivisionByZero)?;
            
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
    pub max_price_deviation_bps: u64, // Maximum allowed deviation between feeds in basis points
}

impl EnhancedOracleAggregator {
    /// Get a weighted average price from all valid feeds.
    /// Now `feed_accounts` must contain the AccountInfos for every feed in `self.feeds`.
    pub fn get_price(&mut self, feed_accounts: &[AccountInfo]) -> Result<OraclePrice> {
        let mut valid_prices: Vec<(i64, u64, i64)> = Vec::new();
        let mut total_weight: u64 = 0;
        let clock_ts = Clock::get()?.unix_timestamp;

        // Validate that we have enough feeds
        require!(feed_accounts.len() >= self.minimum_feeds, 
                ErrorCode::InsufficientValidFeeds);
        
        // Process each feed
        for i in 0..self.feeds.len() {
            // Get thresholds for this feed, using safe defaults if not specified
            let feed_conf_threshold = self.confidence_thresholds.get(i).copied().unwrap_or(u64::MAX);
            let feed_staleness_thresh = self.staleness_thresholds.get(i).copied()
                .unwrap_or(MAX_ORACLE_STALENESS);
            let feed_dev_threshold = self.deviation_thresholds.get(i).copied().unwrap_or(u64::MAX);
            let feed_weight = self.weights.get(i).copied().unwrap_or(0);
            
            // Skip feeds with zero weight
            if feed_weight == 0 {
                continue;
            }

            // Find the AccountInfo for this feed in the provided slice.
            let feed_account = feed_accounts
                .iter()
                .find(|acc| *acc.key == self.feeds[i].feed_pubkey)
                .ok_or(ErrorCode::MissingOracleAccount)?;
                
            // Verify that this account is owned by the Pyth program.
            require_keys_eq!(
                feed_account.owner, 
                PYTH_PROGRAM_ID, 
                ErrorCode::InvalidOwnerForPythAccount
            );

            // Read the price data
            let ephemeral_price = read_pyth_ephemeral(feed_account, &self.feeds[i].kind)?;
            
            // Skip feeds with high confidence (lower is better in Pyth)
            if ephemeral_price.confidence > feed_conf_threshold {
                continue;
            }
            
            // Skip stale feeds
            let age = clock_ts.saturating_sub(ephemeral_price.timestamp);
            if age > feed_staleness_thresh {
                continue;
            }
            
            // Skip negative price feeds for Price type
            if let FeedKind::Price = self.feeds[i].kind {
                if ephemeral_price.price < 0 {
                    continue;
                }
            }
            
            // Check for excessive deviation from previous price
            let prev_price = self.feeds[i].previous_price_i64;
            if prev_price != 0 {
                let diff = (ephemeral_price.price - prev_price).abs() as u64;
                if diff > feed_dev_threshold {
                    continue;
                }
            }
            
            // This feed passed all validations
            valid_prices.push((
                ephemeral_price.price, 
                feed_weight, 
                ephemeral_price.timestamp
            ));
        }

        // Ensure we have enough valid feeds
        require!(
            valid_prices.len() >= self.minimum_feeds, 
            ErrorCode::InsufficientValidFeeds
        );

        // Calculate total weight of valid feeds
        for (_, weight, _) in &valid_prices {
            total_weight = total_weight.saturating_add(*weight);
        }
        require!(total_weight > 0, ErrorCode::InvalidPrice);

        // Check for excessive deviation between feeds if we have multiple valid feeds
        if valid_prices.len() > 1 && self.max_price_deviation_bps > 0 {
            let min_price = valid_prices.iter()
                .map(|(price, _, _)| *price)
                .min()
                .unwrap();
                
            let max_price = valid_prices.iter()
                .map(|(price, _, _)| *price)
                .max()
                .unwrap();
                
            // Skip if either price is non-positive to avoid division issues
            if min_price > 0 && max_price > 0 {
                let deviation_bps = ((max_price - min_price) * 10000) / min_price;
                require!(
                    deviation_bps as u64 <= self.max_price_deviation_bps,
                    ErrorCode::ExcessivePriceDeviation
                );
            }
        }

        // Calculate weighted average price
        let mut sum_price: i128 = 0;
        let mut max_ts: i64 = 0;
        
        for (price, weight, timestamp) in &valid_prices {
            let price_i128 = *price as i128;
            let weight_i128 = *weight as i128;
            
            let partial = price_i128.checked_mul(weight_i128)
                .ok_or(ErrorCode::Overflow)?;
                
            sum_price = sum_price.checked_add(partial)
                .ok_or(ErrorCode::Overflow)?;
                
            max_ts = max(max_ts, *timestamp);
        }
        
        let avg = sum_price.checked_div(total_weight as i128)
            .ok_or(ErrorCode::DivisionByZero)?;
            
        // Enforce a reasonable price range
        require!(
            avg >= -2_000_000_000 && avg <= 2_000_000_000, 
            ErrorCode::InvalidPriceRange
        );

        // Update the aggregator's stored previous values
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

        // Return the final price with metadata
        Ok(OraclePrice {
            price: avg as i64,
            confidence: 0, // We're not computing an aggregate confidence
            timestamp: max_ts,
            status: PriceStatus::Trading,
        })
    }
}

/// Read Pyth price feed data from an AccountInfo.
/// This function uses the pyth_sdk_solana crate to deserialize the on‑chain data.
fn read_pyth_ephemeral(account_info: &AccountInfo, kind: &FeedKind) -> Result<OraclePrice> {
    let data = account_info.data.borrow();
    let price_account = load_price_account(&data)
        .map_err(|_| ErrorCode::InvalidOwnerForPythAccount)?;

    // Pyth SDK stores these fields in `price_account.agg`.
    let publish_time = price_account.timestamp;
    let expo = price_account.exponent;
    let raw_price = price_account.agg.price;
    let raw_conf = price_account.agg.confidence;
    
    // Validate the exponent to prevent overflow
    validate_exponent(expo)?;

    // Safely adjust the raw price based on the exponent
    let adjusted_price = if expo < 0 {
        let factor = 10i64.pow((-expo) as u32);
        raw_price.checked_mul(factor).ok_or(ErrorCode::Overflow)?
    } else if expo > 0 {
        let factor = 10i64.pow(expo as u32);
        raw_price.checked_div(factor).ok_or(ErrorCode::DivisionByZero)?
    } else {
        raw_price
    };

    // Safely adjust the confidence similarly
    let adjusted_conf = if expo < 0 {
        let factor = 10u64.pow((-expo) as u32);
        raw_conf.checked_mul(factor).ok_or(ErrorCode::Overflow)?
    } else if expo > 0 {
        let factor = 10u64.pow(expo as u32);
        raw_conf.checked_div(factor).ok_or(ErrorCode::DivisionByZero)?
    } else {
        raw_conf
    };

    // Determine status based on age
    let clock_ts = Clock::get()?.unix_timestamp;
    let status = if clock_ts.saturating_sub(publish_time) > MAX_ORACLE_STALENESS {
        PriceStatus::Halted
    } else {
        PriceStatus::Trading
    };

    // Return the parsed price
    Ok(OraclePrice {
        price: adjusted_price,
        confidence: adjusted_conf,
        timestamp: publish_time,
        status,
    })
}

// ====================================================
// ADAPTIVE CIRCUIT BREAKER & GOVERNANCE STRUCTS
// ====================================================

#[derive(Clone, AnchorSerialize, AnchorDeserialize)]
pub struct AdaptiveCircuitBreaker {
    pub base_threshold: u64,
    pub volume_threshold: u64,
    pub market_state: MarketState,
    pub last_breach_timestamp: i64,
    pub cooldown_period: i64,
    pub recovery_threshold: u64,
}

#[derive(Clone, Copy, AnchorSerialize, AnchorDeserialize, PartialEq)]
pub enum MarketState {
    Normal,
    Recovering,
    Emergency,
}

impl AdaptiveCircuitBreaker {
    pub fn validate_price_update(
        &self, 
        current_price: u64, 
        new_price: u64, 
        volume: u64
    ) -> Result<()> {
        // Check current market state
        if self.market_state == MarketState::Emergency {
            return err!(ErrorCode::MarketEmergency);
        }
        
        // Skip validation if current price is 0 (initial state)
        if current_price == 0 {
            return Ok(());
        }
        
        // Calculate price change in basis points
        let change_bp = ((new_price as i128) - (current_price as i128))
            .checked_mul(10000)
            .ok_or(ErrorCode::Overflow)?
            .checked_div(current_price as i128)
            .ok_or(ErrorCode::DivisionByZero)?;
            
        let abs_change_bp = change_bp.abs() as u64;
        
        // Apply appropriate threshold based on market state
        let threshold = if self.market_state == MarketState::Recovering {
            self.recovery_threshold
        } else {
            self.base_threshold
        };
        
        // Check if price change exceeds threshold
        require!(abs_change_bp <= threshold, ErrorCode::CircuitBreaker);
        
        // Check minimum volume requirement
        let min_volume = if self.market_state == MarketState::Recovering {
            // Higher volume requirement in recovery mode
            self.volume_threshold.saturating_mul(2)
        } else {
            self.volume_threshold
        };
        
        require!(volume >= min_volume, ErrorCode::InsufficientVolume);
        
        Ok(())
    }
    
    pub fn can_exit_emergency(&self, current_timestamp: i64) -> bool {
        current_timestamp >= self.last_breach_timestamp + self.cooldown_period
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
        // Check if user is the initial governor, which has admin privileges
        if *user == self.initial_governor {
            return required_role != Role::Treasury; // Governor can do anything except Treasury
        }
        
        // Otherwise look up the user's role
        if let Some(current_role) = self.authorities.get(user) {
            match (current_role, &required_role) {
                (Role::Admin, _) => true, // Admin can do anything
                (Role::Operator, Role::Operator | Role::PricePusher) => true,
                (Role::PricePusher, Role::PricePusher) => true,
                (Role::Treasury, Role::Treasury) => true,
                _ => false,
            }
        } else {
            false
        }
        pub fn with_max_size(max_size: usize) -> Self {
            Self {
                data: HashMap::new(),
                max_size: Some(max_size),
            }
        }
        
        pub fn get(&self, key: &K) -> Option<&V> {
            self.data.get(key)
        }
        
        pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
            self.data.get_mut(key)
        }
        
        pub fn insert(&mut self, key: K, value: V) -> Result<()> {
            // Check if we're at capacity
            if let Some(max) = self.max_size {
                if self.data.len() >= max && !self.data.contains_key(&key) {
                    return err!(ErrorCode::MapFull);
                }
            }
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
        
        pub fn is_full(&self) -> bool {
            if let Some(max) = self.max_size {
                self.data.len() >= max
            } else {
                false
            }
        }
    }

    /// Verifiable state struct with hash for state verification
#[derive(Clone, AnchorSerialize, AnchorDeserialize)]
pub struct VerifiableState {
    pub dummy: u8,
    pub state_hash: [u8; 32], // Hash of critical state for verification
    pub last_hash_update: i64,
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
    pub last_emergency_action: i64,
    pub protocol_version: u8, // Added for future upgradability
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
        
        let min_interval = info.min_interval.max(10); // Minimum 10 seconds between pushes
        
        if now - info.last_push < min_interval {
            return err!(ErrorCode::RateLimitExceeded);
        }
        
        Ok(())
    }
    
    pub fn update_state_hash(&mut self) -> Result<()> {
        // Create a hash of critical state components
        let mut hasher = anchor_lang::solana_program::hash::Hasher::default();
        
        // Add key state variables to the hash
        hasher.hash(self.authority.as_ref());
        hasher.hash(self.governance_authority.as_ref());
        hasher.hash(&self.mint_price_bull.to_le_bytes());
        hasher.hash(&self.mint_price_bear.to_le_bytes());
        hasher.hash(&[self.is_market_open as u8]);
        
        // Current time
        let now = Clock::get()?.unix_timestamp;
        hasher.hash(&now.to_le_bytes());
        
        // Store hash and update timestamp
        let result = hasher.result();
        self.ver_state.state_hash.copy_from_slice(result.as_ref());
        self.ver_state.last_hash_update = now;
        
        Ok(())
    }
    
    pub fn verify_state_hash(&self) -> Result<()> {
        // Only verify if hash has been set
        if self.ver_state.last_hash_update == 0 {
            return Ok(());
        }
        
        // Create verification hash
        let mut hasher = anchor_lang::solana_program::hash::Hasher::default();
        
        hasher.hash(self.authority.as_ref());
        hasher.hash(self.governance_authority.as_ref());
        hasher.hash(&self.mint_price_bull.to_le_bytes());
        hasher.hash(&self.mint_price_bear.to_le_bytes());
        hasher.hash(&[self.is_market_open as u8]);
        
        // Use the stored timestamp for verification
        hasher.hash(&self.ver_state.last_hash_update.to_le_bytes());
        
        let verification = hasher.result();
        
        // Verify the hash matches
        let stored_hash = self.ver_state.state_hash;
        require!(
            verification.as_ref() == stored_hash,
            ErrorCode::InvalidStateHash
        );
        
        Ok(())
    }
}
// ====================================================
// ACCOUNT CONTEXTS
// ====================================================

#[derive(Accounts)]
pub struct InitializeState<'info> {
    #[account(
        init,
        payer = authority,
        space = MagaIndexState::MAX_SIZE,
        constraint = reward_treasury.data_is_empty() @ ErrorCode::TreasuryAlreadyInitialized
    )]
    pub maga_index_state: Account<'info, MagaIndexState>,
    
    #[account(mut)]
    pub authority: Signer<'info>,
    
    #[account(mut)]
    pub reward_treasury: Signer<'info>,
    
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct MintFractionBatch<'info> {
    #[account(
        mut,
        constraint = maga_index_state.is_market_open @ ErrorCode::MarketClosed
    )]
    pub maga_index_state: Account<'info, MagaIndexState>,
    
    #[account(
        init,
        payer = payer,
        space = 8 + FractionAccount::MAX_SIZE,
        seeds = [b"fraction", payer.key().as_ref()],
        bump,
    )]
    pub fraction_account: Account<'info, FractionAccount>,
    
    #[account(mut)]
    pub payer: Signer<'info>,
    
    #[account(
        mut,
        constraint = reward_treasury.key() == maga_index_state.reward_treasury @ ErrorCode::InvalidRewardTreasury
    )]
    /// CHECK: This account receives SOL
    pub reward_treasury: AccountInfo<'info>,
    
    pub system_program: Program<'info, System>,
    
    #[account(address = clock::ID)]
    pub clock_sysvar: Sysvar<'info, Clock>,
}

#[derive(Accounts)]
pub struct PerformUpkeepBatch<'info> {
    #[account(mut)]
    pub maga_index_state: Account<'info, MagaIndexState>,
    
    #[account(
        mut,
        constraint = maga_index_state.gov.is_authorized(&authority.key(), Role::Operator) @ ErrorCode::Unauthorized
    )]
    pub authority: Signer<'info>,
    
    #[account(address = clock::ID)]
    pub clock_sysvar: Sysvar<'info, Clock>,
}

#[derive(Accounts)]
pub struct PerformUpkeep<'info> {
    #[account(mut)]
    pub maga_index_state: Account<'info, MagaIndexState>,
    
    #[account(
        mut,
        constraint = maga_index_state.gov.is_authorized(&authority.key(), Role::Operator) @ ErrorCode::Unauthorized
    )]
    pub authority: Signer<'info>,
    
    #[account(
        mut,
        constraint = reward_treasury.key() == maga_index_state.reward_treasury @ ErrorCode::InvalidRewardTreasury
    )]
    /// CHECK: This account sends SOL
    pub reward_treasury: AccountInfo<'info>,
    
    #[account(mut)]
    pub nft_mint: Account<'info, Mint>,
    
    #[account(
        mut,
        seeds = [b"MAGA_INDEX_REWARD_POOL", program_id().as_ref()],
        bump,
    )]
    /// CHECK: The mint authority (PDA)
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
    
    #[account(
        mut,
        constraint = maga_index_state.gov.is_authorized(&authority.key(), Role::Admin) @ ErrorCode::Unauthorized
    )]
    pub authority: Signer<'info>,
    
    #[account(address = clock::ID)]
    pub clock_sysvar: Sysvar<'info, Clock>,
}

#[derive(Accounts)]
pub struct PerformYearlyGlitchAuction<'info> {
    #[account(mut)]
    pub maga_index_state: Account<'info, MagaIndexState>,
    
    #[account(mut)]
    pub payer: Signer<'info>,
    
    #[account(
        init,
        payer = payer,
        associated_token::mint = glitch_mint,
        associated_token::authority = top_user_account
    )]
    pub user_ata: Account<'info, TokenAccount>,
    
    pub glitch_mint: Account<'info, Mint>,
    
    /// CHECK: The top user
    pub top_user_account: AccountInfo<'info>,
    
    /// CHECK: The mint authority (PDA)
    #[account(
        mut,
        seeds = [b"MAGA_INDEX_REWARD_POOL", program_id().as_ref()],
        bump,
    )]
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

/// In the improved version, we require validation of Pyth feed accounts
#[derive(Accounts)]
pub struct PushPrice<'info> {
    #[account(
        mut,
        constraint = maga_index_state.is_market_open @ ErrorCode::MarketClosed,
        constraint = maga_index_state.gov.is_authorized(&price_pusher.key(), Role::PricePusher) @ ErrorCode::Unauthorized
    )]
    pub maga_index_state: Account<'info, MagaIndexState>,
    
    #[account(mut)]
    pub price_pusher: Signer<'info>,
    
    #[account(address = clock::ID)]
    pub clock_sysvar: Sysvar<'info, Clock>,
    
    // All Pyth feed accounts are expected to be passed as remaining_accounts.
}

#[derive(Accounts)]
pub struct ManagePricePusher<'info> {
    #[account(mut)]
    pub maga_index_state: Account<'info, MagaIndexState>,
    
    #[account(
        mut,
        constraint = maga_index_state.gov.is_authorized(&authority.key(), Role::Admin) @ ErrorCode::Unauthorized
    )]
    pub authority: Signer<'info>,
    
    #[account(address = clock::ID)]
    pub clock_sysvar: Sysvar<'info, Clock>,
}

// Additional helper functions

/// Safe transfer function with validation
pub fn transfer_from_treasury<'info>(
    treasury_account: &AccountInfo<'info>,
    system_program: &AccountInfo<'info>,
    recipient: Pubkey,
    lamports: u64,
) -> Result<()> {
    // Ensure we don't drain the treasury below rent-exempt minimum
    let min_balance = Rent::get()?.minimum_balance(0);
    let current_balance = treasury_account.lamports();
    
    require!(
        current_balance >= lamports.saturating_add(min_balance),
        ErrorCode::InsufficientFunds
    );
    
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

/// Additional user position utility functions
pub fn user_bull_value(position: &UserPosition, day: u64) -> u64 {
    position.recent_entries
        .iter()
        .filter(|e| e.is_bull)
        .map(|e| calculate_decayed_fraction_value(e, day))
        .sum()
}

pub fn user_bear_value(position: &UserPosition, day: u64) -> u64 {
    position.recent_entries
        .iter()
        .filter(|e| !e.is_bull)
        .map(|e| calculate_decayed_fraction_value(e, day))
        .sum()
}

pub fn find_second_place(
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
#[program]
pub mod maga_index_advanced_plus_upgrade {
    use super::*;

    /// Initialize the full state with security checks
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
        
        // Input validation
        require!(timelock_duration >= 0, ErrorCode::InvalidParameter);
        require!(min_valid_price > 0, ErrorCode::InvalidParameter);
        require!(twap_window > 0, ErrorCode::InvalidParameter);
        require!(max_history_size > 0 && max_history_size <= 100, ErrorCode::InvalidParameter);
        
        // Ensure the aggregator has at least one feed
        require!(!aggregator_conf.feeds.is_empty(), ErrorCode::InvalidParameter);
        require!(aggregator_conf.minimum_feeds > 0, ErrorCode::InvalidParameter);
        
        let state = &mut ctx.accounts.maga_index_state;
        
        // Set basic state
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
        
        // Initialize with first price point
        let clock = Clock::get()?;
        let bull_price = state.mint_price_bull;
        state.price_history.push(clock.unix_timestamp, bull_price, 1)?;
        state.last_oracle_price = state.mint_price_bull as i64;
        
        // Initialize collections with size limits
        state.yearly_totals = OptimizedVecMap::with_max_size(10); // Max 10 years
        state.user_positions = OptimizedVecMap::with_max_size(MAX_USER_POSITIONS);
        state.settled_days = OptimizedVecMap::with_max_size(400); // Max ~1 year of days
        state.value_checkpoints = OptimizedVecMap::with_max_size(100);
        
        state.price_conf = PriceConfig {
            weighting_spy: 1000,
            weighting_btc: 0,
            volume_weight: 0,
            max_deviation_per_update: 20000,
            max_total_deviation: 50000,
            min_history_points: 5,
            min_bull_price: min_valid_price,
            min_bear_price: min_valid_price,
            staleness_threshold: 300, // 5 minutes staleness by default
        };
        
        state.price_pushers = OptimizedVecMap::with_max_size(20); // Limit price pushers
        state.batch_offset = 0;
        state.batch_in_progress_day = None;
        state.creator_wallet = ctx.accounts.reward_treasury.key();
        state.last_emergency_action = 0;
        state.protocol_version = 1;
        
        // Generate initial state hash
        state.update_state_hash()?;
        
        emit!(PriceUpdated {
            bull_price: state.mint_price_bull,
            bear_price: state.mint_price_bear,
            oracle_price: state.last_oracle_price,
            timestamp: clock.unix_timestamp,
        });
        
        Ok(())
    }

    /// Mint fraction batch with improved security and validation
    pub fn mint_fraction_batch(
        ctx: Context<MintFractionBatch>,
        day: u64,
        is_bull: bool,
        quantity: u64,
        max_price: u64,
        deadline: i64,
    ) -> Result<()> {
        msg!("Instruction: MintFractionBatch");
        
        // Get current time
        let clock = Clock::get()?;
        let current_ts = clock.unix_timestamp;
        let current_day = current_ts / 86400;
        
        // State validation
        let state = &mut ctx.accounts.maga_index_state;
        state.verify_state_hash()?;
        require!(state.is_market_open, ErrorCode::MarketClosed);
        require!(current_ts < TERM_END_TIMESTAMP, ErrorCode::TermEnded);
        
        // Input validation
        require!(day as i64 == current_day, ErrorCode::InvalidDay);
        require!(deadline >= current_ts + 120, ErrorCode::DeadlineTooSoon);
        require!(current_ts <= deadline, ErrorCode::Expired);
        require!(quantity > 0, ErrorCode::ZeroQuantity);
        require!(quantity <= 1_000_000, ErrorCode::ExcessiveQuantity); // Cap max quantity
        
        // Get the latest oracle price
        let final_oracle_price = state.oracle_aggregator.get_price(ctx.remaining_accounts)?;
        
        // Calculate new price
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
        
        // Verify price doesn't trigger circuit breaker
        state.circuit_breaker.validate_price_update(current_price, calculated_price, volume)?;
        
        // Enforce price limits
        require!(calculated_price <= max_price, ErrorCode::PriceTooHigh);
        
        // Calculate total cost
        let required_total = calculated_price.checked_mul(quantity)
            .ok_or(ErrorCode::Overflow)?;
            
        // Perform the payment transfer
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
        
        // Update fraction account
        ctx.accounts.fraction_account.owner = ctx.accounts.payer.key();
        ctx.accounts.fraction_account.bump = *ctx.bumps.get("fraction_account").unwrap();
        
        // Get or create user position
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
            
        // Create new entry
        let new_entry = FractionEntry {
            day,
            quantity,
            initial_value: quantity,
            is_bull,
        };
        
        // Update position
        user_position.recent_entries.push_back(new_entry);
        user_position.raw_total_fractions = user_position
            .raw_total_fractions
            .checked_add(quantity)
            .ok_or(ErrorCode::Overflow)?;
            
        // Update fraction account
        ctx.accounts.fraction_account.total_contributed = ctx
            .accounts
            .fraction_account
            .total_contributed
            .checked_add(required_total)
            .ok_or(ErrorCode::Overflow)?;
        ctx.accounts.fraction_account.last_update_timestamp = current_ts;
        ctx.accounts.fraction_account.is_active = true;
        
        // Store updated user position
        state.user_positions.insert(ctx.accounts.fraction_account.owner, user_position)?;
        
        // Update price history
        state.price_history.push(current_ts, calculated_price, volume)?;
        
        // Update current prices
        if is_bull {
            state.mint_price_bull = calculated_price;
        } else {
            state.mint_price_bear = calculated_price;
        }
        
        // Store the last oracle price
        state.last_oracle_price = final_oracle_price.price;
        
        // Update state hash
        state.update_state_hash()?;
        
        // Emit event
        emit!(FractionMinted {
            user: ctx.accounts.payer.key(),
            day,
            is_bull,
            quantity,
            price: calculated_price,
            timestamp: current_ts,
        });
        
        Ok(())
    }

    /// Perform partial batch for day "day" over at most max_users with improved security
    pub fn perform_upkeep_batch(
        ctx: Context<PerformUpkeepBatch>, 
        day: u64, 
        max_users: u64
    ) -> Result<()> {
        msg!("Instruction: PerformUpkeepBatch");
        
        let state = &mut ctx.accounts.maga_index_state;
        
        // Verify state hash
        state.verify_state_hash()?;
        
        // Authorization check
        require!(
            state.gov.is_authorized(&ctx.accounts.authority.key(), Role::Operator),
            ErrorCode::Unauthorized
        );
        
        // Current day validation
        let current_day = Clock::get()?.unix_timestamp / 86400;
        
        // Only process days that have ended
        require!(day as i64 < current_day, ErrorCode::InvalidDay);
        
        // Initialize batch or validate ongoing batch
        if state.batch_in_progress_day.is_none() {
            state.batch_in_progress_day = Some(day);
            state.batch_offset = 0;
        } else {
            let in_prog_day = state.batch_in_progress_day.unwrap();
            require_eq!(in_prog_day, day, ErrorCode::InvalidDay);
        }
        
        // Get all user keys in sorted order for deterministic processing
        let mut user_keys: Vec<Pubkey> = state.user_positions.data.keys().copied().collect();
        user_keys.sort();
        
        let total_users = user_keys.len() as u64;
        
        // Check if batch is already done
        if state.batch_offset >= total_users {
            return Ok(());
        }
        
        // Cap max users per batch to prevent excessive compute
        let max_users_capped = min(max_users, 100);
        let end_index = state.batch_offset
            .saturating_add(max_

                .saturating_add(max_users_capped)
            .min(total_users);
            
        // Process each user in the batch
        for i in state.batch_offset..end_index {
            let user_pk = user_keys[i as usize];
            if let Some(pos) = state.user_positions.get_mut(&user_pk) {
                pos.consolidate_if_needed(day);
            }
        }
        
        // Update batch offset
        state.batch_offset = end_index;
        
        // Update state hash
        state.update_state_hash()?;
        
        Ok(())
    }

    /// Final step to distribute lamports/mint NFT once batch is done with improved security
    pub fn perform_upkeep(ctx: Context<PerformUpkeep>, day: u64) -> Result<()> {
        msg!("Instruction: PerformUpkeep");
        
        let state = &mut ctx.accounts.maga_index_state;
        
        // Verify state hash
        state.verify_state_hash()?;
        
        // Authorization check
        require!(
            state.gov.is_authorized(&ctx.accounts.authority.key(), Role::Operator),
            ErrorCode::Unauthorized
        );
        
        // Validate this day hasn't been settled
        let already_settled = state.settled_days.get(&day).copied().unwrap_or(false);
        require!(!already_settled, ErrorCode::DayAlreadySettled);
        
        // Current day validation
        let current_ts = Clock::get()?.unix_timestamp;
        let current_day = current_ts / 86400;
        require!(day as i64 < current_day, ErrorCode::InvalidDay);
        
        // Verify batch is complete for this day
        require!(state.batch_in_progress_day == Some(day), ErrorCode::InvalidDay);
        let total_users = state.user_positions.data.len() as u64;
        require!(state.batch_offset >= total_users, ErrorCode::DayNotYetSettledInBatch);

        // Reset batch processing state
        state.batch_in_progress_day = None;
        state.batch_offset = 0;
        
        // Process rewards and determine winners
        let (maybe_winner, maybe_second, side_accounts, is_bullish_day) = {
            // Get latest oracle price to determine if day was bullish or bearish
            let final_oracle_price = state.oracle_aggregator.get_price(ctx.remaining_accounts)?;
            let is_bullish_day_local = final_oracle_price.price > state.last_oracle_price;
            
            // Find winners and distribute rewards
            let maybe_winner_local = distribute_daily_rewards(state, day, is_bullish_day_local)?;
            let maybe_second_local = find_second_place(state, day, is_bullish_day_local);
            let side_accounts_local = find_winning_side_accounts(state, day, is_bullish_day_local);
            
            // Mark day as settled
            state.settled_days.insert(day, true)?;
            
            // Update yearly aggregator
            update_yearly_aggregator(state, day)?;
            
            // Reset prices to minimum values
            state.mint_price_bull = state.price_conf.min_bull_price;
            state.mint_price_bear = state.price_conf.min_bear_price;
            
            // Prune old data
            let oldest_day_to_keep = day.saturating_sub(365);
            state.value_checkpoints.prune_below(&oldest_day_to_keep);
            state.settled_days.prune_below(&oldest_day_to_keep);
            
            // Return the values
            (maybe_winner_local, maybe_second_local, side_accounts_local, is_bullish_day_local)
        };

        // Distribute rewards from treasury
        let reward_treasury_balance = ctx.accounts.reward_treasury.lamports();
        let rent_exempt_reserve = Rent::get()?.minimum_balance(0);
        let distributable = reward_treasury_balance.saturating_sub(rent_exempt_reserve);

        if distributable > 0 {
            // Calculate shares with safe math
            let creator_cut = safe_math::mul_div(distributable, 70, 100)?;
            let winner_cut = safe_math::mul_div(distributable, 15, 100)?;
            let second_cut = safe_math::mul_div(distributable, 10, 100)?;
            let side_cut = safe_math::mul_div(distributable, 5, 100)?;
            
            // Transfer to creator
            if creator_cut > 0 {
                transfer_from_treasury(
                    &ctx.accounts.reward_treasury,
                    &ctx.accounts.system_program.to_account_info(),
                    state.creator_wallet,
                    creator_cut
                )?;
            }
            
            // Transfer to first place winner
            if let Some(winner_pk) = maybe_winner {
                if winner_cut > 0 {
                    transfer_from_treasury(
                        &ctx.accounts.reward_treasury,
                        &ctx.accounts.system_program.to_account_info(),
                        winner_pk,
                        winner_cut
                    )?;
                }
            }
            
            // Transfer to second place
            if let Some(second_pk) = maybe_second {
                if second_cut > 0 {
                    transfer_from_treasury(
                        &ctx.accounts.reward_treasury,
                        &ctx.accounts.system_program.to_account_info(),
                        second_pk,
                        second_cut
                    )?;
                }
            }
            
            // Distribute to winning side
            if side_cut > 0 && !side_accounts.is_empty() {
                let side_total = {
                    let st = &ctx.accounts.maga_index_state;
                    sum_side_value(st, &side_accounts, day, is_bullish_day)
                };
                
                if side_total > 0 {
                    for user_pk in &side_accounts {
                        let user_val = {
                            let st = &ctx.accounts.maga_index_state;
                            user_value(st, *user_pk, day, is_bullish_day)
                        };
                        
                        // Calculate proportional share with safe math
                        let share = safe_math::mul_div(side_cut, user_val, side_total)?;
                        
                        if share > 0 {
                            transfer_from_treasury(
                                &ctx.accounts.reward_treasury,
                                &ctx.accounts.system_program.to_account_info(),
                                *user_pk,
                                share
                            )?;
                        }
                    }
                }
            }
            
            // Emit event for reward distribution
            emit!(RewardDistributed {
                day,
                winner: maybe_winner,
                runner_up: maybe_second,
                total_amount: distributable,
                timestamp: current_ts,
            });
        }

        // Mint NFT for the winner if one exists
        if let Some(winner_pubkey) = maybe_winner {
            mint_daily_artwork(&ctx, winner_pubkey, day)?;
        }
        
        // Update state hash
        state.update_state_hash()?;
        
        Ok(())
    }

    /// Perform a yearly glitch auction with improved security
    pub fn perform_yearly_glitch_auction(
        ctx: Context<PerformYearlyGlitchAuction>,
        _top_user: Pubkey,
        year: u64,
        metadata_uri_input: String,
    ) -> Result<()> {
        msg!("Instruction: PerformYearlyGlitchAuction");
        
        // Validate year is allowed
        require!(
            matches!(year, 2026 | 2027 | 2028 | 2029), 
            ErrorCode::InvalidYearForAuction
        );
        
        // Verify the metadata URI is reasonable
        require!(
            metadata_uri_input.starts_with("https://") && 
            metadata_uri_input.len() <= 200,
            ErrorCode::InvalidMetadataUri
        );
        
        // Check if auction is active
        let ts = Clock::get()?.unix_timestamp;
        let start_date = match year {
            2026 => GLITCH_AUCTION_DATE_2026,
            2027 => GLITCH_AUCTION_DATE_2027,
            2028 => GLITCH_AUCTION_DATE_2028,
            2029 => GLITCH_AUCTION_DATE_2029,
            _ => return err!(ErrorCode::InvalidYearForAuction),
        };
        
        // Validate timestamp is within auction window
        require!(
            ts >= start_date && ts < start_date + 86400, 
            ErrorCode::AuctionNotActive
        );
        
        // Find the true top user (don't rely on input parameter)
        let top_user = get_top_user_by_raw_fractions(&ctx.accounts.maga_index_state)
            .ok_or(ErrorCode::NoUserFound)?;
            
        // Validate top user matches account provided
        require_keys_eq!(
            top_user, 
            ctx.accounts.top_user_account.key(), 
            ErrorCode::InvalidTopUser
        );
        
        // Derive mint authority PDA
        let (mint_authority_pda, bump) = MagaIndexState::get_mint_authority_address(&ctx.program_id);
        
        // Validate mint authority
        require_keys_eq!(
            mint_authority_pda,
            ctx.accounts.mint_authority.key(),
            ErrorCode::InvalidMintAuthority
        );
        
        // Setup PDA signing
        let signer_seeds: &[&[u8]] = &[
            b"MAGA_INDEX_REWARD_POOL",
            ctx.program_id.as_ref(),
            &[bump],
        ];
        let signer_seeds_array = [signer_seeds];
        
        // Derive metadata accounts
        let metadata_program_id = MPL_TOKEN_METADATA_ID;
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
        
        // Setup NFT metadata
        let name = format!("Glitch NFT - Year {}", year);
        let symbol = "GLITCH".to_string();
        
        // Create metadata instruction
        let ix_metadata = create_metadata_accounts_v3(
            metadata_program_id,
            metadata_pda,
            glitch_mint_key,
            mint_authority_pda,
            ctx.accounts.payer.key(),
            mint_authority_pda,
            name,
            symbol,
            metadata_uri_input,
            Some(vec![Creator {
                address: mint_authority_pda,
                verified: true,
                share: 100,
            }]),
            500, // 5% royalty
            true, // Is mutable
            true, // Update authority is signer
            None, // Collection
            None, // Uses
            None, // Collection Details
        );
        
        // Execute metadata creation
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
        
        // Mint the token
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
        
        // Create master edition
        let ix_master = create_master_edition_v3(
            metadata_program_id,
            master_edition_pda,
            glitch_mint_key,
            mint_authority_pda,
            ctx.accounts.payer.key(),
            metadata_pda,
            ctx.accounts.token_program.key(),
            Some(0) // Max supply of 0 means no edition prints allowed
        );
        
        // Execute master edition creation
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
        
        // Reset all user positions for new year
        reset_all_user_positions(&mut ctx.accounts.maga_index_state);
        
        // Emit NFT minted event
        emit!(NFTMinted {
            recipient: top_user,
            mint: glitch_mint_key,
            day: year as u64 * 365, // Use year converted to days as identifier
            timestamp: ts,
        });
        
        Ok(())
    }

    /// Register a retroactive NFT with improved security
    pub fn register_retroactive_nft(
        ctx: Context<RegisterRetroactiveNFT>,
        day: u64,
        mint: Pubkey,
        metadata_uri: String,
    ) -> Result<()> {
        msg!("Instruction: RegisterRetroactiveNFT");
        
        // Verify state hash
        ctx.accounts.maga_index_state.verify_state_hash()?;
        
        // Check admin permission
        require!(
            ctx.accounts.maga_index_state.gov.is_authorized(
                &ctx.accounts.authority.key(), 
                Role::Admin
            ),
            ErrorCode::Unauthorized
        );
        
        // Validate metadata URI
        require!(
            metadata_uri.starts_with("https://") && 
            metadata_uri.len() <= 200,
            ErrorCode::InvalidMetadataUri
        );
        
        // Validate day
        let current_day = Clock::get()?.unix_timestamp / 86400;
        require!(day as i64 < current_day, ErrorCode::InvalidDay);
        
        // Could store this information in the state if needed
        
        Ok(())
    }

    /// Push a price update from Pyth with improved security
    pub fn push_price_update(ctx: Context<PushPrice>) -> Result<()> {
        msg!("Instruction: PushPriceUpdate");
        
        let state = &mut ctx.accounts.maga_index_state;
        
        // Verify state hash
        state.verify_state_hash()?;
        
        // Market state check
        require!(state.is_market_open, ErrorCode::MarketClosed);
        
        // Authorization check
        require!(
            state.gov.is_authorized(&ctx.accounts.price_pusher.key(), Role::PricePusher), 
            ErrorCode::Unauthorized
        );
        
        // Ensure each remaining account is owned by the Pyth program
        for acc in ctx.remaining_accounts.iter() {
            require_keys_eq!(
                acc.owner, 
                PYTH_PROGRAM_ID, 
                ErrorCode::InvalidOwnerForPythAccount
            );
        }
        
        // Rate limiting check
        let now = Clock::get()?.unix_timestamp;
        state.validate_price_push(&ctx.accounts.price_pusher.key(), now)?;
        
        // Pass all remaining accounts (the Pyth feed accounts) into the aggregator
        let final_oracle_price = state.oracle_aggregator.get_price(ctx.remaining_accounts)?;
        
        // Apply volume for price update
        let volume = 1;
        
        // Set default previous price if none exists
        let prev_price = if state.last_oracle_price == 0 {
            final_oracle_price.price
        } else {
            state.last_oracle_price
        };
        
        // Calculate new prices
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
        
        // Validate against circuit breaker
        state.circuit_breaker.validate_price_update(
            state.mint_price_bull, 
            new_bull_price, 
            volume
        )?;
        
        state.circuit_breaker.validate_price_update(
            state.mint_price_bear, 
            new_bear_price, 
            volume
        )?;
        
        // Update prices
        state.mint_price_bull = new_bull_price;
        state.mint_price_bear = new_bear_price;
        state.last_oracle_price = final_oracle_price.price;

        // Update pusher info
        let mut pusher_info = state
            .price_pushers
            .get(&ctx.accounts.price_pusher.key())
            .cloned()
            .unwrap_or_else(|| PricePusherInfo {
                last_push: 0,
                push_count: 0,
                is_active: true,
                min_interval: 10,
            });
            
        pusher_info.last_push = now;
        pusher_info.push_count = pusher_info.push_count.saturating_add(1);
        
        state.price_pushers.insert(ctx.accounts.price_pusher.key(), pusher_info)?;
        
        // Update price history
        state.price_history.push(now, new_bull_price, volume)?;
        
        // Update state hash
        state.update_state_hash()?;
        
        // Emit price updated event
        emit!(PriceUpdated {
            bull_price: new_bull_price,
            bear_price: new_bear_price,
            oracle_price: final_oracle_price.price,
            timestamp: now,
        });
        
        Ok(())
    }

    /// Manage Price Pusher - add with improved security
    pub fn add_price_pusher(ctx: Context<ManagePricePusher>, pusher: Pubkey) -> Result<()> {
        msg!("Instruction: AddPricePusher");
        
        let state = &mut ctx.accounts.maga_index_state;
        
        // Verify state hash
        state.verify_state_hash()?;
        
        // Admin authorization check
        require!(
            state.gov.is_authorized(&ctx.accounts.authority.key(), Role::Admin), 
            ErrorCode::Unauthorized
        );
        
        // Check if pusher already exists
        require!(
            state.price_pushers.get(&pusher).is_none(), 
            ErrorCode::PusherAlreadyExists
        );
        
        // Check timelock for admin action
        let now = Clock::get()?.unix_timestamp;
        let gov_action = GovernanceAction::AddPricePusher(pusher);
        state.gov.check_timelock(&gov_action, now, state.last_timelock_set)?;
        
        // Create new pusher info with reasonable defaults
        let info = PricePusherInfo { 
            last_push: 0, 
            push_count: 0, 
            is_active: true,
            min_interval: 10,
        };
        
        // Add to price pushers map
        state.price_pushers.insert(pusher, info)?;
        
        // Add role in governance
        state.gov.authorities.insert(pusher, Role::PricePusher)?;
        
        // Update state hash
        state.update_state_hash()?;
        
        Ok(())
    }

    /// Manage Price Pusher - remove with improved security
    pub fn remove_price_pusher(ctx: Context<ManagePricePusher>, pusher: Pubkey) -> Result<()> {
        msg!("Instruction: RemovePricePusher");
        
        let state = &mut ctx.accounts.maga_index_state;
        
        // Verify state hash
        state.verify_state_hash()?;
        
        // Admin authorization check
        require!(
            state.gov.is_authorized(&ctx.accounts.authority.key(), Role::Admin), 
            ErrorCode::Unauthorized
        );
        
        // Check timelock for admin action
        let now = Clock::get()?.unix_timestamp;
        let gov_action = GovernanceAction::RemovePricePusher(pusher);
        state.gov.check_timelock(&gov_action, now, state.last_timelock_set)?;
        
        // Remove from both maps
        state.price_pushers.remove(&pusher);
        state.gov.authorities.remove(&pusher);
        
        // Update state hash
        state.update_state_hash()?;
        
        Ok(())
    }

    /// Manage Price Pusher - update status with improved security
    pub fn update_pusher_status(
        ctx: Context<ManagePricePusher>, 
        pusher: Pubkey, 
        is_active: bool
    ) -> Result<()> {
        msg!("Instruction: UpdatePusherStatus");
        
        let state = &mut ctx.accounts.maga_index_state;
        
        // Verify state hash
        state.verify_state_hash()?;
        
        // Operator authorization check
        require!(
            state.gov.is_authorized(&ctx.accounts.authority.key(), Role::Operator), 
            ErrorCode::Unauthorized
        );
        
        // Get existing pusher info
        let mut info = state.price_pushers.get(&pusher)
            .cloned()
            .ok_or(ErrorCode::PusherNotFound)?;
            
        // Update status
        info.is_active = is_active;
        
        // Store updated info
        state.price_pushers.insert(pusher, info)?;
        
        // Update state hash
        state.update_state_hash()?;
        
        Ok(())
    }
    
    /// New function: Update circuit breaker configuration
    pub fn update_circuit_breaker(
        ctx: Context<ManagePricePusher>,
        new_config: AdaptiveCircuitBreaker
    ) -> Result<()> {
        msg!("Instruction: UpdateCircuitBreaker");
        
        let state = &mut ctx.accounts.maga_index_state;
        
        // Verify state hash
        state.verify_state_hash()?;
        
        // Admin authorization check
        require!(
            state.gov.is_authorized(&ctx.accounts.authority.key(), Role::Admin), 
            ErrorCode::Unauthorized
        );
        
        // Check timelock for admin action
        let now = Clock::get()?.unix_timestamp;
        let gov_action = GovernanceAction::UpdateCircuitBreaker(new_config.clone());
        state.gov.check_timelock(&gov_action, now, state.last_timelock_set)?;
        
        // Input validation for new config
        require!(new_config.base_threshold > 0, ErrorCode::InvalidParameter);
        require!(new_config.volume_threshold > 0, ErrorCode::InvalidParameter);
        require!(new_config.recovery_threshold > 0, ErrorCode::InvalidParameter);
        
        // Update circuit breaker config
        state.circuit_breaker = new_config;
        
        // Update timelock timestamp
        state.last_timelock_set = now;
        
        // Update state hash
        state.update_state_hash()?;
        
        Ok(())
    }
    
    /// New function: Set timelock duration
    pub fn set_timelock_duration(
        ctx: Context<ManagePricePusher>,
        new_duration: i64
    ) -> Result<()> {
        msg!("Instruction: SetTimelockDuration");
        
        let state = &mut ctx.accounts.maga_index_state;
        
        // Verify state hash
        state.verify_state_hash()?;
        
        // Admin authorization check
        require!(
            state.gov.is_authorized(&ctx.accounts.authority.key(), Role::Admin), 
            ErrorCode::Unauthorized
        );
        
        // Input validation
        require!(new_duration >= 0, ErrorCode::InvalidParameter);
        
        // Check timelock for admin action
        let now = Clock::get()?.unix_timestamp;
        let gov_action = GovernanceAction::SetTimelock(new_duration);
        state.gov.check_timelock(&gov_action, now, state.last_timelock_set)?;
        
        // Update timelock duration
        state.timelock_duration = new_duration;
        state.last_timelock_set = now;
        
        // Update state hash
        state.update_state_hash()?;
        
        Ok(())
    }
    
    /// New function: Emergency market pause
    pub fn emergency_pause(ctx: Context<ManagePricePusher>) -> Result<()> {
        msg!("Instruction: EmergencyPause");
        
        let state = &mut ctx.accounts.maga_index_state;
        
        // Verify state hash
        state.verify_state_hash()?;
        
        // Admin authorization check
        require!(
            state.gov.is_authorized(&ctx.accounts.authority.key(), Role::Admin), 
            ErrorCode::Unauthorized
        );
        
        // Check if market is already closed
        require!(state.is_market_open, ErrorCode::MarketAlreadyClosed);
        
        // Close the market
        state.is_market_open = false;
        
        // Record emergency action timestamp
        state.last_emergency_action = Clock::get()?.unix_timestamp;
        
        // Update circuit breaker state
        state.circuit_breaker.market_state = MarketState::Emergency;
        state.circuit_breaker.last_breach_timestamp = state.last_emergency_action;
        
        // Update state hash
        state.update_state_hash()?;
        
        Ok(())
    }
    
    /// New function: Resume market after emergency
    pub fn resume_market(ctx: Context<ManagePricePusher>) -> Result<()> {
        msg!("Instruction: ResumeMarket");
        
        let state = &mut ctx.accounts.maga_index_state;
        
        // Verify state hash
        state.verify_state_hash()?;
        
        // Admin authorization check
        require!(
            state.gov.is_authorized(&ctx.accounts.authority.key(), Role::Admin), 
            ErrorCode::Unauthorized
        );
        
        // Check if market is already open
        require!(!state.is_market_open, ErrorCode::MarketAlreadyOpen);
        
        // Check cooldown period
        let now = Clock::get()?.unix_timestamp;
        require!(
            state.circuit_breaker.can_exit_emergency(now),
            ErrorCode::EmergencyCooldownNotExpired
        );
        
        // Open the market
        state.is_market_open = true;
        
        // Update circuit breaker state
        state.circuit_breaker.market_state = MarketState::Recovering;
        
        // Update state hash
        state.update_state_hash()?;
        
        Ok(())
    }
}
.saturating_add(max_users_capped)
            .min(total_users);
            
        // Process each user in the batch
        for i in state.batch_offset..end_index {
            let user_pk = user_keys[i as usize];
            if let Some(pos) = state.user_positions.get_mut(&user_pk) {
                pos.consolidate_if_needed(day);
            }
        }
        
        // Update batch offset
        state.batch_offset = end_index;
        
        // Update state hash
        state.update_state_hash()?;
        
        Ok(())
    }

    /// Final step to distribute lamports/mint NFT once batch is done with improved security
    pub fn perform_upkeep(ctx: Context<PerformUpkeep>, day: u64) -> Result<()> {
        msg!("Instruction: PerformUpkeep");
        
        let state = &mut ctx.accounts.maga_index_state;
        
        // Verify state hash
        state.verify_state_hash()?;
        
        // Authorization check
        require!(
            state.gov.is_authorized(&ctx.accounts.authority.key(), Role::Operator),
            ErrorCode::Unauthorized
        );
        
        // Validate this day hasn't been settled
        let already_settled = state.settled_days.get(&day).copied().unwrap_or(false);
        require!(!already_settled, ErrorCode::DayAlreadySettled);
        
        // Current day validation
        let current_ts = Clock::get()?.unix_timestamp;
        let current_day = current_ts / 86400;
        require!(day as i64 < current_day, ErrorCode::InvalidDay);
        
        // Verify batch is complete for this day
        require!(state.batch_in_progress_day == Some(day), ErrorCode::InvalidDay);
        let total_users = state.user_positions.data.len() as u64;
        require!(state.batch_offset >= total_users, ErrorCode::DayNotYetSettledInBatch);

        // Reset batch processing state
        state.batch_in_progress_day = None;
        state.batch_offset = 0;
        
        // Process rewards and determine winners
        let (maybe_winner, maybe_second, side_accounts, is_bullish_day) = {
            // Get latest oracle price to determine if day was bullish or bearish
            let final_oracle_price = state.oracle_aggregator.get_price(ctx.remaining_accounts)?;
            let is_bullish_day_local = final_oracle_price.price > state.last_oracle_price;
            
            // Find winners and distribute rewards
            let maybe_winner_local = distribute_daily_rewards(state, day, is_bullish_day_local)?;
            let maybe_second_local = find_second_place(state, day, is_bullish_day_local);
            let side_accounts_local = find_winning_side_accounts(state, day, is_bullish_day_local);
            
            // Mark day as settled
            state.settled_days.insert(day, true)?;
            
            // Update yearly aggregator
            update_yearly_aggregator(state, day)?;
            
            // Reset prices to minimum values
            state.mint_price_bull = state.price_conf.min_bull_price;
            state.mint_price_bear = state.price_conf.min_bear_price;
            
            // Prune old data
            let oldest_day_to_keep = day.saturating_sub(365);
            state.value_checkpoints.prune_below(&oldest_day_to_keep);
            state.settled_days.prune_below(&oldest_day_to_keep);
            
            // Return the values
            (maybe_winner_local, maybe_second_local, side_accounts_local, is_bullish_day_local)
        };

        // Distribute rewards from treasury
        let reward_treasury_balance = ctx.accounts.reward_treasury.lamports();
        let rent_exempt_reserve = Rent::get()?.minimum_balance(0);
        let distributable = reward_treasury_balance.saturating_sub(rent_exempt_reserve);

        if distributable > 0 {
            // Calculate shares with safe math
            let creator_cut = safe_math::mul_div(distributable, 70, 100)?;
            let winner_cut = safe_math::mul_div(distributable, 15, 100)?;
            let second_cut = safe_math::mul_div(distributable, 10, 100)?;
            let side_cut = safe_math::mul_div(distributable, 5, 100)?;
            
            // Transfer to creator
            if creator_cut > 0 {
                transfer_from_treasury(
                    &ctx.accounts.reward_treasury,
                    &ctx.accounts.system_program.to_account_info(),
                    state.creator_wallet,
                    creator_cut
                )?;
            }
            
            // Transfer to first place winner
            if let Some(winner_pk) = maybe_winner {
                if winner_cut > 0 {
                    transfer_from_treasury(
                        &ctx.accounts.reward_treasury,
                        &ctx.accounts.system_program.to_account_info(),
                        winner_pk,
                        winner_cut
                    )?;
                }
            }
            
            // Transfer to second place
            if let Some(second_pk) = maybe_second {
                if second_cut > 0 {
                    transfer_from_treasury(
                        &ctx.accounts.reward_treasury,
                        &ctx.accounts.system_program.to_account_info(),
                        second_pk,
                        second_cut
                    )?;
                }
            }
            
            // Distribute to winning side
            if side_cut > 0 && !side_accounts.is_empty() {
                let side_total = {
                    let st = &ctx.accounts.maga_index_state;
                    sum_side_value(st, &side_accounts, day, is_bullish_day)
                };
                
                if side_total > 0 {
                    for user_pk in &side_accounts {
                        let user_val = {
                            let st = &ctx.accounts.maga_index_state;
                            user_value(st, *user_pk, day, is_bullish_day)
                        };
                        
                        // Calculate proportional share with safe math
                        let share = safe_math::mul_div(side_cut, user_val, side_total)?;
                        
                        if share > 0 {
                            transfer_from_treasury(
                                &ctx.accounts.reward_treasury,
                                &ctx.accounts.system_program.to_account_info(),
                                *user_pk,
                                share
                            )?;
                        }
                    }
                }
            }
            
            // Emit event for reward distribution
            emit!(RewardDistributed {
                day,
                winner: maybe_winner,
                runner_up: maybe_second,
                total_amount: distributable,
                timestamp: current_ts,
            });
        }

        // Mint NFT for the winner if one exists
        if let Some(winner_pubkey) = maybe_winner {
            mint_daily_artwork(&ctx, winner_pubkey, day)?;
        }
        
        // Update state hash
        state.update_state_hash()?;
        
        Ok(())
    }

    /// Perform a yearly glitch auction with improved security
    pub fn perform_yearly_glitch_auction(
        ctx: Context<PerformYearlyGlitchAuction>,
        _top_user: Pubkey,
        year: u64,
        metadata_uri_input: String,
    ) -> Result<()> {
        msg!("Instruction: PerformYearlyGlitchAuction");
        
        // Validate year is allowed
        require!(
            matches!(year, 2026 | 2027 | 2028 | 2029), 
            ErrorCode::InvalidYearForAuction
        );
        
        // Verify the metadata URI is reasonable
        require!(
            metadata_uri_input.starts_with("https://") && 
            metadata_uri_input.len() <= 200,
            ErrorCode::InvalidMetadataUri
        );
        
        // Check if auction is active
        let ts = Clock::get()?.unix_timestamp;
        let start_date = match year {
            2026 => GLITCH_AUCTION_DATE_2026,
            2027 => GLITCH_AUCTION_DATE_2027,
            2028 => GLITCH_AUCTION_DATE_2028,
            2029 => GLITCH_AUCTION_DATE_2029,
            _ => return err!(ErrorCode::InvalidYearForAuction),
        };
        
        // Validate timestamp is within auction window
        require!(
            ts >= start_date && ts < start_date + 86400, 
            ErrorCode::AuctionNotActive
        );
        
        // Find the true top user (don't rely on input parameter)
        let top_user = get_top_user_by_raw_fractions(&ctx.accounts.maga_index_state)
            .ok_or(ErrorCode::NoUserFound)?;
            
        // Validate top user matches account provided
        require_keys_eq!(
            top_user, 
            ctx.accounts.top_user_account.key(), 
            ErrorCode::InvalidTopUser
        );
        
        // Derive mint authority PDA
        let (mint_authority_pda, bump) = MagaIndexState::get_mint_authority_address(&ctx.program_id);
        
        // Validate mint authority
        require_keys_eq!(
            mint_authority_pda,
            ctx.accounts.mint_authority.key(),
            ErrorCode::InvalidMintAuthority
        );
        
        // Setup PDA signing
        let signer_seeds: &[&[u8]] = &[
            b"MAGA_INDEX_REWARD_POOL",
            ctx.program_id.as_ref(),
            &[bump],
        ];
        let signer_seeds_array = [signer_seeds];
        
        // Derive metadata accounts
        let metadata_program_id = MPL_TOKEN_METADATA_ID;
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
        
        // Setup NFT metadata
        let name = format!("Glitch NFT - Year {}", year);
        let symbol = "GLITCH".to_string();
        
        // Create metadata instruction
        let ix_metadata = create_metadata_accounts_v3(
            metadata_program_id,
            metadata_pda,
            glitch_mint_key,
            mint_authority_pda,
            ctx.accounts.payer.key(),
            mint_authority_pda,
            name,
            symbol,
            metadata_uri_input,
            Some(vec![Creator {
                address: mint_authority_pda,
                verified: true,
                share: 100,
            }]),
            500, // 5% royalty
            true, // Is mutable
            true, // Update authority is signer
            None, // Collection
            None, // Uses
            None, // Collection Details
        );
        
        // Execute metadata creation
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
        
        // Mint the token
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
        
        // Create master edition
        let ix_master = create_master_edition_v3(
            metadata_program_id,
            master_edition_pda,
            glitch_mint_key,
            mint_authority_pda,
            ctx.accounts.payer.key(),
            metadata_pda,
            ctx.accounts.token_program.key(),
            Some(0) // Max supply of 0 means no edition prints allowed
        );
        
        // Execute master edition creation
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
        
        // Reset all user positions for new year
        reset_all_user_positions(&mut ctx.accounts.maga_index_state);
        
        // Emit NFT minted event
        emit!(NFTMinted {
            recipient: top_user,
            mint: glitch_mint_key,
            day: year as u64 * 365, // Use year converted to days as identifier
            timestamp: ts,
        });
        
        Ok(())
    }

    /// Register a retroactive NFT with improved security
    pub fn register_retroactive_nft(
        ctx: Context<RegisterRetroactiveNFT>,
        day: u64,
        mint: Pubkey,
        metadata_uri: String,
    ) -> Result<()> {
        msg!("Instruction: RegisterRetroactiveNFT");
        
        // Verify state hash
        ctx.accounts.maga_index_state.verify_state_hash()?;
        
        // Check admin permission
        require!(
            ctx.accounts.maga_index_state.gov.is_authorized(
                &ctx.accounts.authority.key(), 
                Role::Admin
            ),
            ErrorCode::Unauthorized
        );
        
        // Validate metadata URI
        require!(
            metadata_uri.starts_with("https://") && 
            metadata_uri.len() <= 200,
            ErrorCode::InvalidMetadataUri
        );
        
        // Validate day
        let current_day = Clock::get()?.unix_timestamp / 86400;
        require!(day as i64 < current_day, ErrorCode::InvalidDay);
        
        // Could store this information in the state if needed
        
        Ok(())
    }

    /// Push a price update from Pyth with improved security
    pub fn push_price_update(ctx: Context<PushPrice>) -> Result<()> {
        msg!("Instruction: PushPriceUpdate");
        
        let state = &mut ctx.accounts.maga_index_state;
        
        // Verify state hash
        state.verify_state_hash()?;
        
        // Market state check
        require!(state.is_market_open, ErrorCode::MarketClosed);
        
        // Authorization check
        require!(
            state.gov.is_authorized(&ctx.accounts.price_pusher.key(), Role::PricePusher), 
            ErrorCode::Unauthorized
        );
        
        // Ensure each remaining account is owned by the Pyth program
        for acc in ctx.remaining_accounts.iter() {
            require_keys_eq!(
                acc.owner, 
                PYTH_PROGRAM_ID, 
                ErrorCode::InvalidOwnerForPythAccount
            );
        }
        
        // Rate limiting check
        let now = Clock::get()?.unix_timestamp;
        state.validate_price_push(&ctx.accounts.price_pusher.key(), now)?;
        
        // Pass all remaining accounts (the Pyth feed accounts) into the aggregator
        let final_oracle_price = state.oracle_aggregator.get_price(ctx.remaining_accounts)?;
        
        // Apply volume for price update
        let volume = 1;
        
        // Set default previous price if none exists
        let prev_price = if state.last_oracle_price == 0 {
            final_oracle_price.price
        } else {
            state.last_oracle_price
        };
        
        // Calculate new prices
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
        
        // Validate against circuit breaker
        state.circuit_breaker.validate_price_update(
            state.mint_price_bull, 
            new_bull_price, 
            volume
        )?;
        
        state.circuit_breaker.validate_price_update(
            state.mint_price_bear, 
            new_bear_price, 
            volume
        )?;
        
        // Update prices
        state.mint_price_bull = new_bull_price;
        state.mint_price_bear = new_bear_price;
        state.last_oracle_price = final_oracle_price.price;

        // Update pusher info
        let mut pusher_info = state
            .price_pushers
            .get(&ctx.accounts.price_pusher.key())
            .cloned()
            .unwrap_or_else(|| PricePusherInfo {
                last_push: 0,
                push_count: 0,
                is_active: true,
                min_interval: 10,
            });
            
        pusher_info.last_push = now;
        pusher_info.push_count = pusher_info.push_count.saturating_add(1);
        
        state.price_pushers.insert(ctx.accounts.price_pusher.key(), pusher_info)?;
        
        // Update price history
        state.price_history.push(now, new_bull_price, volume)?;
        
        // Update state hash
        state.update_state_hash()?;
        
        // Emit price updated event
        emit!(PriceUpdated {
            bull_price: new_bull_price,
            bear_price: new_bear_price,
            oracle_price: final_oracle_price.price,
            timestamp: now,
        });
        
        Ok(())
    }

    /// Manage Price Pusher - add with improved security
    pub fn add_price_pusher(ctx: Context<ManagePricePusher>, pusher: Pubkey) -> Result<()> {
        msg!("Instruction: AddPricePusher");
        
        let state = &mut ctx.accounts.maga_index_state;
        
        // Verify state hash
        state.verify_state_hash()?;
        
        // Admin authorization check
        require!(
            state.gov.is_authorized(&ctx.accounts.authority.key(), Role::Admin), 
            ErrorCode::Unauthorized
        );
        
        // Check if pusher already exists
        require!(
            state.price_pushers.get(&pusher).is_none(), 
            ErrorCode::PusherAlreadyExists
        );
        
        // Check timelock for admin action
        let now = Clock::get()?.unix_timestamp;
        let gov_action = GovernanceAction::AddPricePusher(pusher);
        state.gov.check_timelock(&gov_action, now, state.last_timelock_set)?;
        
        // Create new pusher info with reasonable defaults
        let info = PricePusherInfo { 
            last_push: 0, 
            push_count: 0, 
            is_active: true,
            min_interval: 10,
        };
        
        // Add to price pushers map
        state.price_pushers.insert(pusher, info)?;
        
        // Add role in governance
        state.gov.authorities.insert(pusher, Role::PricePusher)?;
        
        // Update state hash
        state.update_state_hash()?;
        
        Ok(())
    }

    /// Manage Price Pusher - remove with improved security
    pub fn remove_price_pusher(ctx: Context<ManagePricePusher>, pusher: Pubkey) -> Result<()> {
        msg!("Instruction: RemovePricePusher");
        
        let state = &mut ctx.accounts.maga_index_state;
        
        // Verify state hash
        state.verify_state_hash()?;
        
        // Admin authorization check
        require!(
            state.gov.is_authorized(&ctx.accounts.authority.key(), Role::Admin), 
            ErrorCode::Unauthorized
        );
        
        // Check timelock for admin action
        let now = Clock::get()?.unix_timestamp;
        let gov_action = GovernanceAction::RemovePricePusher(pusher);
        state.gov.check_timelock(&gov_action, now, state.last_timelock_set)?;
        
        // Remove from both maps
        state.price_pushers.remove(&pusher);
        state.gov.authorities.remove(&pusher);
        
        // Update state hash
        state.update_state_hash()?;
        
        Ok(())
    }

    /// Manage Price Pusher - update status with improved security
    pub fn update_pusher_status(
        ctx: Context<ManagePricePusher>, 
        pusher: Pubkey, 
        is_active: bool
    ) -> Result<()> {
        msg!("Instruction: UpdatePusherStatus");
        
        let state = &mut ctx.accounts.maga_index_state;
        
        // Verify state hash
        state.verify_state_hash()?;
        
        // Operator authorization check
        require!(
            state.gov.is_authorized(&ctx.accounts.authority.key(), Role::Operator), 
            ErrorCode::Unauthorized
        );
        
        // Get existing pusher info
        let mut info = state.price_pushers.get(&pusher)
            .cloned()
            .ok_or(ErrorCode::PusherNotFound)?;
            
        // Update status
        info.is_active = is_active;
        
        // Store updated info
        state.price_pushers.insert(pusher, info)?;
        
        // Update state hash
        state.update_state_hash()?;
        
        Ok(())
    }
    
    /// New function: Update circuit breaker configuration
    pub fn update_circuit_breaker(
        ctx: Context<ManagePricePusher>,
        new_config: AdaptiveCircuitBreaker
    ) -> Result<()> {
        msg!("Instruction: UpdateCircuitBreaker");
        
        let state = &mut ctx.accounts.maga_index_state;
        
        // Verify state hash
        state.verify_state_hash()?;
        
        // Admin authorization check
        require!(
            state.gov.is_authorized(&ctx.accounts.authority.key(), Role::Admin), 
            ErrorCode::Unauthorized
        );
        
        // Check timelock for admin action
        let now = Clock::get()?.unix_timestamp;
        let gov_action = GovernanceAction::UpdateCircuitBreaker(new_config.clone());
        state.gov.check_timelock(&gov_action, now, state.last_timelock_set)?;
        
        // Input validation for new config
        require!(new_config.base_threshold > 0, ErrorCode::InvalidParameter);
        require!(new_config.volume_threshold > 0, ErrorCode::InvalidParameter);
        require!(new_config.recovery_threshold > 0, ErrorCode::InvalidParameter);
        
        // Update circuit breaker config
        state.circuit_breaker = new_config;
        
        // Update timelock timestamp
        state.last_timelock_set = now;
        
        // Update state hash
        state.update_state_hash()?;
        
        Ok(())
    }
    
    /// New function: Set timelock duration
    pub fn set_timelock_duration(
        ctx: Context<ManagePricePusher>,
        new_duration: i64
    ) -> Result<()> {
        msg!("Instruction: SetTimelockDuration");
        
        let state = &mut ctx.accounts.maga_index_state;
        
        // Verify state hash
        state.verify_state_hash()?;
        
        // Admin authorization check
        require!(
            state.gov.is_authorized(&ctx.accounts.authority.key(), Role::Admin), 
            ErrorCode::Unauthorized
        );
        
        // Input validation
        require!(new_duration >= 0, ErrorCode::InvalidParameter);
        
        // Check timelock for admin action
        let now = Clock::get()?.unix_timestamp;
        let gov_action = GovernanceAction::SetTimelock(new_duration);
        state.gov.check_timelock(&gov_action, now, state.last_timelock_set)?;
        
        // Update timelock duration
        state.timelock_duration = new_duration;
        state.last_timelock_set = now;
        
        // Update state hash
        state.update_state_hash()?;
        
        Ok(())
    }
    
    /// New function: Emergency market pause
    pub fn emergency_pause(ctx: Context<ManagePricePusher>) -> Result<()> {
        msg!("Instruction: EmergencyPause");
        
        let state = &mut ctx.accounts.maga_index_state;
        
        // Verify state hash
        state.verify_state_hash()?;
        
        // Admin authorization check
        require!(
            state.gov.is_authorized(&ctx.accounts.authority.key(), Role::Admin), 
            ErrorCode::Unauthorized
        );
        
        // Check if market is already closed
        require!(state.is_market_open, ErrorCode::MarketAlreadyClosed);
        
        // Close the market
        state.is_market_open = false;
        
        // Record emergency action timestamp
        state.last_emergency_action = Clock::get()?.unix_timestamp;
        
        // Update circuit breaker state
        state.circuit_breaker.market_state = MarketState::Emergency;
        state.circuit_breaker.last_breach_timestamp = state.last_emergency_action;
        
        // Update state hash
        state.update_state_hash()?;
        
        Ok(())
    }
    
    /// New function: Resume market after emergency
    pub fn resume_market(ctx: Context<ManagePricePusher>) -> Result<()> {
        msg!("Instruction: ResumeMarket");
        
        let state = &mut ctx.accounts.maga_index_state;
        
        // Verify state hash
        state.verify_state_hash()?;
        
        // Admin authorization check
        require!(
            state.gov.is_authorized(&ctx.accounts.authority.key(), Role::Admin), 
            ErrorCode::Unauthorized
        );
        
        // Check if market is already open
        require!(!state.is_market_open, ErrorCode::MarketAlreadyOpen);
        
        // Check cooldown period
        let now = Clock::get()?.unix_timestamp;
        require!(
            state.circuit_breaker.can_exit_emergency(now),
            ErrorCode::EmergencyCooldownNotExpired
        );
        
        // Open the market
        state.is_market_open = true;
        
        // Update circuit breaker state
        state.circuit_breaker.market_state = MarketState::Recovering;
        
        // Update state hash
        state.update_state_hash()?;
        
        Ok(())
    }
}pub fn user_value(
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

pub fn sum_side_value(
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
// ERROR CODES
// ====================================================

#[error_code]
pub enum ErrorCode {
    #[msg("Unauthorized access")]
    Unauthorized,
    
    #[msg("Arithmetic overflow")]
    Overflow,
    
    #[msg("Division by zero")]
    DivisionByZero,
    
    #[msg("Invalid timestamp order")]
    InvalidTimeOrder,
    
    #[msg("Market is closed")]
    MarketClosed,
    
    #[msg("Market is already closed")]
    MarketAlreadyClosed,
    
    #[msg("Market is already open")]
    MarketAlreadyOpen,
    
    #[msg("Market is in emergency state")]
    MarketEmergency,
    
    #[msg("Emergency cooldown period not expired")]
    EmergencyCooldownNotExpired,
    
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
    
    #[msg("Excessive quantity")]
    ExcessiveQuantity,
    
    #[msg("Excessive volatility")]
    ExcessiveVolatility,
    
    #[msg("Excessive price deviation between feeds")]
    ExcessivePriceDeviation,
    
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
    
    #[msg("Invalid parameter")]
    InvalidParameter,
    
    #[msg("Invalid exponent")]
    InvalidExponent,
    
    #[msg("Invalid percentage")]
    InvalidPercentage,
    
    #[msg("Treasury already initialized")]
    TreasuryAlreadyInitialized,
    
    #[msg("Invalid reward treasury")]
    InvalidRewardTreasury,
    
    #[msg("Insufficient data for calculation")]
    InsufficientData,
    
    #[msg("Invalid metadata URI")]
    InvalidMetadataUri,
    
    #[msg("Invalid mint authority")]
    InvalidMintAuthority,
    
    #[msg("Invalid top user")]
    InvalidTopUser,
    
    #[msg("Insufficient funds")]
    InsufficientFunds,
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
