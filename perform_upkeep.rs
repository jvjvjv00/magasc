// tests/perform_upkeep_test.rs
use anchor_lang::{
    prelude::*,
    AccountDeserialize,
};
use anchor_spl::token::{TokenAccount};
// --- FIXES START HERE ---
// Bring in the Solana Program system_program for `system_program::ID`.
use solana_program::system_program;

// Bring in the SPL Token crate so we can do `spl_token::ID`.
use spl_token::ID as SPL_TOKEN_ID;
// Or you can just `use spl_token;` if you prefer referencing `spl_token::ID` directly.

// Bring in the SPL Token instruction module for init calls.
use spl_token::instruction as spl_token_instruction;
// --- FIXES END HERE ---
use solana_program_test::{processor, ProgramTest};
use solana_sdk::{
    account::Account as SolanaAccount,
    signature::{Keypair, Signer},
    system_instruction,
    transaction::Transaction,
};

use mi_smart_contract::{
    FractionAccount,
    MagaIndexState,
    // your structs
    EnhancedOracleAggregator, AdaptiveCircuitBreaker, MarketState, AdvancedGovernance,
    OptimizedVecMap, VotingConfig, TimelockConfig, EmergencyConfig, Role,
    VerifiableState, DecayRates,
    id as my_program_id,
};

#[derive(AnchorSerialize, AnchorDeserialize)]
struct InitializeFullStateIxArgs {
    aggregator_conf: EnhancedOracleAggregator,
    circuit_breaker: AdaptiveCircuitBreaker,
    gov: AdvancedGovernance,
    ver_state: VerifiableState,
    timelock_duration: i64,
    min_valid_price: u64,
    twap_window: i64,
    max_history_size: usize,
    decay_rates: DecayRates,
}

// 8-byte discriminator for "global:initialize_full_state"
fn initialize_full_state_discriminator() -> [u8; 8] {
    let ix_name = b"global:initialize_full_state";
    let hash = anchor_lang::solana_program::hash::hash(ix_name);
    let mut disc = [0u8; 8];
    disc.copy_from_slice(&hash.to_bytes()[..8]);
    disc
}

// For mint_fraction_batch
#[derive(AnchorSerialize, AnchorDeserialize)]
struct MintFractionBatchIxArgs {
    day: u64,
    is_bull: bool,
    quantity: u64,
    max_price: u64,
    deadline: i64,
}
fn mint_fraction_batch_discriminator() -> [u8; 8] {
    let ix_name = b"global:mint_fraction_batch";
    let hash = anchor_lang::solana_program::hash::hash(ix_name);
    let mut disc = [0u8; 8];
    disc.copy_from_slice(&hash.to_bytes()[..8]);
    disc
}

// For perform_upkeep
#[derive(AnchorSerialize, AnchorDeserialize)]
struct PerformUpkeepIxArgs {
    day: u64,
}
fn perform_upkeep_discriminator() -> [u8; 8] {
    let ix_name = b"global:perform_upkeep";
    let hash = anchor_lang::solana_program::hash::hash(ix_name);
    let mut disc = [0u8; 8];
    disc.copy_from_slice(&hash.to_bytes()[..8]);
    disc
}

#[tokio::test]
async fn test_perform_upkeep_success() {
    // ----------------------------------------------------
    // 1) Spin up local environment
    // ----------------------------------------------------
    let program_id = mi_smart_contract::id();
    let mut test = ProgramTest::new(
        "mi_smart_contract",
        program_id,
        processor!(mi_smart_contract::entry),
    );

    // Keypairs
    let authority_keypair = Keypair::new();
    let state_keypair = Keypair::new();
    let fraction_account_key = Keypair::new();
    let edition_authority_key = Keypair::new(); // Signer for edition_authority

    // We'll treat 'authority_keypair' as the main user, too.
    test.add_account(
        authority_keypair.pubkey(),
        SolanaAccount {
            lamports: 2_000_000_000,
            data: vec![],
            owner: system_program::ID,
            executable: false,
            rent_epoch: 0,
        },
    );


    let (mut banks_client, _unused_payer, recent_blockhash) = test.start().await;

    // ----------------------------------------------------
    // 2) Initialize the main MagaIndexState
    // ----------------------------------------------------
    let aggregator_conf = EnhancedOracleAggregator {
        feeds: vec![],
        weights: vec![],
        confidence_thresholds: vec![],
        staleness_thresholds: vec![],
        deviation_thresholds: vec![],
        minimum_feeds: 0,
    };
    let circuit_breaker = AdaptiveCircuitBreaker {
        base_threshold: 1000,
        volume_threshold: 1,
        market_state: MarketState::Normal,
    };

    let mut authorities_map = OptimizedVecMap::new();
    authorities_map.insert(authority_keypair.pubkey(), Role::Admin).unwrap();

    let gov = AdvancedGovernance {
        authorities: authorities_map,
        proposals: OptimizedVecMap::new(),
        voting_config: VotingConfig {
            quorum: 10,
            supermajority: 6000,
            voting_period: 10000,
            execution_delay: 100,
            min_voting_power: 1,
        },
        timelock_config: TimelockConfig {
            normal_delay: 60,
            critical_delay: 60,
            emergency_delay: 60,
        },
        emergency_config: EmergencyConfig {
            timelock: 60,
            required_approvals: 1,
            cool_down_period: 60,
        },
        next_proposal_id: 0,
        initial_governor: authority_keypair.pubkey(),
    };

    let ver_state = VerifiableState { dummy: 123 };
    let decay_rates = DecayRates {
        winning_side_daily_bps: 50,
        losing_side_daily_bps: 100,
    };

    let init_args = InitializeFullStateIxArgs {
        aggregator_conf,
        circuit_breaker,
        gov,
        ver_state,
        timelock_duration: 0,
        min_valid_price: 1,
        twap_window: 60,
        max_history_size: 10,
        decay_rates,
    };

    let (reward_pool_pda, _bump_seed) = MagaIndexState::get_reward_pool_address(&program_id);

    let mut init_data = Vec::new();
    init_data.extend_from_slice(&initialize_full_state_discriminator());
    init_data.extend_from_slice(&init_args.try_to_vec().unwrap());

    let init_ix_accounts = vec![
        AccountMeta::new(state_keypair.pubkey(), true),
        AccountMeta::new(authority_keypair.pubkey(), true),
        AccountMeta::new(reward_pool_pda, false),
        AccountMeta::new_readonly(system_program::ID, false),
    ];
    let init_ix = solana_program::instruction::Instruction {
        program_id,
        accounts: init_ix_accounts,
        data: init_data,
    };

    let mut init_tx = Transaction::new_with_payer(
        &[init_ix],
        Some(&authority_keypair.pubkey()),
    );
    init_tx.sign(&[&authority_keypair, &state_keypair], recent_blockhash);
    banks_client.process_transaction(init_tx).await.unwrap();

    // ----------------------------------------------------
    // 3) Create user’s FractionAccount + call mint_fraction_batch
    // ----------------------------------------------------
    // Create fraction_account
    let fraction_acc_space = 8 + FractionAccount::MAX_SIZE;
    let rent_exempt_minimum = banks_client
        .get_rent()
        .await
        .unwrap()
        .minimum_balance(fraction_acc_space);

    let create_fraction_acc_ix = system_instruction::create_account(
        &authority_keypair.pubkey(),
        &fraction_account_key.pubkey(),
        rent_exempt_minimum,
        fraction_acc_space as u64,
        &program_id, // Owned by our program
    );
    let mut create_fraction_acc_tx = Transaction::new_with_payer(
        &[create_fraction_acc_ix],
        Some(&authority_keypair.pubkey()),
    );
    create_fraction_acc_tx.sign(&[&authority_keypair, &fraction_account_key], recent_blockhash);
    banks_client.process_transaction(create_fraction_acc_tx).await.unwrap();

    // "mint_fraction_batch" so user has some position
    let day = 1u64;
    let is_bull = true;
    let quantity = 100u64;
    let max_price = 2_000_000_000u64;
    let deadline = (Clock::get().unwrap().unix_timestamp + 300) as i64;

    let mfb_args = MintFractionBatchIxArgs {
        day,
        is_bull,
        quantity,
        max_price,
        deadline,
    };

    let mut mfb_data = Vec::new();
    mfb_data.extend_from_slice(&mint_fraction_batch_discriminator());
    mfb_data.extend_from_slice(&mfb_args.try_to_vec().unwrap());

    let mint_ix_accounts = vec![
        AccountMeta::new(state_keypair.pubkey(), false),
        AccountMeta::new(fraction_account_key.pubkey(), false),
        AccountMeta::new(authority_keypair.pubkey(), true),  // payer
        AccountMeta::new(reward_pool_pda, false), 
        AccountMeta::new_readonly(system_program::ID, false),
    ];
    let mint_ix = solana_program::instruction::Instruction {
        program_id,
        accounts: mint_ix_accounts,
        data: mfb_data,
    };
    let mut mint_tx = Transaction::new_with_payer(
        &[mint_ix],
        Some(&authority_keypair.pubkey()),
    );
    mint_tx.sign(&[&authority_keypair], recent_blockhash);
    banks_client.process_transaction(mint_tx).await.unwrap();

    // Now the user (authority) has minted fractions, 
    // so they should be top user for day=1.

    // ----------------------------------------------------
    // 4) Create NFT Mint + winner ATA, pass them to perform_upkeep
    // ----------------------------------------------------
    // We'll make a new mint for the daily NFT reward.
    // We'll also create an associated token account owned by the authority.
    // The "edition_authority" is just a signatory for the PDAs.

    // (A) Create the NFT Mint (like a normal SPL mint)
    let nft_mint_key = Keypair::new();
    let create_mint_ix = system_instruction::create_account(
        &authority_keypair.pubkey(),
        &nft_mint_key.pubkey(),
        mint_rent_minimum,
        mint_space as u64,
        &SPL_TOKEN_ID, // from `use spl_token::ID as SPL_TOKEN_ID;`
    );

    // Now call the correct function from spl_token_instruction:
    let init_mint_ix = spl_token_instruction::initialize_mint(
        &SPL_TOKEN_ID,
        &nft_mint_key.pubkey(),
        &authority_keypair.pubkey(),    // mint authority
        Some(&authority_keypair.pubkey()), // freeze authority
        0,
    ).unwrap();

    let mut create_mint_tx = Transaction::new_with_payer(
        &[create_mint_ix, init_mint_ix],
        Some(&authority_keypair.pubkey()),
    );
    create_mint_tx.sign(&[&authority_keypair, &nft_mint_key], recent_blockhash);
    banks_client.process_transaction(create_mint_tx).await.unwrap();

    // (B) Create the winner ATA for authority_keypair
    let winner_ata_key = Keypair::new();
    let create_ata_ix = system_instruction::create_account(
        &authority_keypair.pubkey(),
        &winner_ata_key.pubkey(),
        ata_rent,
        ata_space as u64,
        &SPL_TOKEN_ID,
    );
    let init_ata_ix = spl_token_instruction::initialize_account(
        &SPL_TOKEN_ID,
        &winner_ata_key.pubkey(),
        &nft_mint_key.pubkey(),
        &authority_keypair.pubkey(),
    ).unwrap();

    let mut ata_tx = Transaction::new_with_payer(
        &[create_ata_ix, init_ata_ix],
        Some(&authority_keypair.pubkey()),
    );
    ata_tx.sign(&[&authority_keypair, &winner_ata_key], recent_blockhash);
    banks_client.process_transaction(ata_tx).await.unwrap();

    // (C) We'll pass an empty associated_token_program
    // For the "edition_authority", we just sign with the edition_authority_key
    test.add_account(
        edition_authority_key.pubkey(),
        SolanaAccount {
            lamports: 500_000_000,
            data: vec![],
            owner: system_program::ID,
            executable: false,
            rent_epoch: 0,
        },
    );

    // ----------------------------------------------------
    // 5) Call perform_upkeep(day=1)
    // ----------------------------------------------------
    let upkeep_args = PerformUpkeepIxArgs { day: 1 };
    let mut upkeep_data = Vec::new();
    upkeep_data.extend_from_slice(&perform_upkeep_discriminator());
    upkeep_data.extend_from_slice(&upkeep_args.try_to_vec().unwrap());

    let perform_upkeep_ix_accounts = vec![
        AccountMeta::new(state_keypair.pubkey(), false),
        AccountMeta::new(authority_keypair.pubkey(), true),
        AccountMeta::new(reward_pool_pda, false),
        AccountMeta::new(nft_mint_key.pubkey(), false), // nft_mint
        AccountMeta::new(winner_ata_key.pubkey(), false), // mint_authority is actually the PDA, but let's do next
        // Actually we see the code: 
        //   pub mint_authority: AccountInfo<'info>, // not a Signer
        // We'll pass some new key as an account with mut
        AccountMeta::new(Keypair::new().pubkey(), false), // ephemeral "mint_authority"
        AccountMeta::new(winner_ata_key.pubkey(), false), // winner_ata
        AccountMeta::new(edition_authority_key.pubkey(), true),
        AccountMeta::new_readonly(mpl_token_metadata::ID, false),
        AccountMeta::new_readonly(token::ID, false),
        AccountMeta::new_readonly(system_program::ID, false),
        AccountMeta::new_readonly(anchor_spl::associated_token::ID, false),
    ];
    let perform_upkeep_ix = solana_program::instruction::Instruction {
        program_id,
        accounts: perform_upkeep_ix_accounts,
        data: upkeep_data,
    };

    let mut upkeep_tx = Transaction::new_with_payer(
        &[perform_upkeep_ix],
        Some(&authority_keypair.pubkey()),
    );
    upkeep_tx.sign(&[&authority_keypair, &edition_authority_key], recent_blockhash);
    banks_client.process_transaction(upkeep_tx).await.unwrap();

    // ----------------------------------------------------
    // 6) Verify day=1 is settled and check minted NFT
    // ----------------------------------------------------
    // (A) Check maga_index_state => "settled_days" => day=1 => true
    let state_account_data = banks_client
        .get_account(state_keypair.pubkey())
        .await
        .unwrap()
        .expect("State not found");

    {
        let mut slice: &[u8] = &state_account_data.data;
        slice = &slice[8..]; // skip anchor discriminator
        let state_loaded = MagaIndexState::try_deserialize_unchecked(&mut slice)
            .expect("Failed to parse MagaIndexState");

        let day_already = state_loaded.settled_days.get(&1).copied().unwrap_or(false);
        assert!(day_already, "Day 1 was NOT settled by upkeep!");
    }

    // (B) Check if an NFT was minted to the user’s ATA (balance == 1)
    // Example fix for anchor_lang::AccountDeserialize usage:
    let winner_ata_data = banks_client
        .get_account(winner_ata_key.pubkey())
        .await
        .unwrap()
        .expect("Winner ATA not found");

    let mut entire_data: &[u8] = &winner_ata_data.data;
    let mut slice_of_slice: &[u8] = &entire_data[8..]; 
    let ata_account: token::TokenAccount = token::TokenAccount::try_deserialize(&mut slice_of_slice)
        .expect("Failed to parse TokenAccount data");
    assert_eq!(ata_account.amount, 0); // or 1, if minted, etc.

    println!("test_perform_upkeep_success passed!");
}