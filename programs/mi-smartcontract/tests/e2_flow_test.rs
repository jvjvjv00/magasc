// programs/mi-smartcontract/tests/e2_flow_test.rs

use anchor_lang::{
    prelude::*,
    AccountDeserialize,
};
use borsh::BorshSerialize;
use solana_program_test::{processor, ProgramTest};
use solana_program::{
    clock::Clock as ProgramClock,
    program_pack::Pack,
    sysvar::{clock, rent, instructions},   // <-- Add these imports
};
use solana_sdk::{
    account::Account as SolanaAccount,
    instruction::AccountMeta,
    pubkey::Pubkey,
    signature::{Keypair, Signer},
    system_program,
    transaction::Transaction,
};

// Direct spl-token references
use spl_token::{
    instruction as token_ix,
    state::{Account as SplTokenAccount, Mint as SplMint},
    id as spl_token_program_id,
};

// Direct spl-associated-token-account references
use spl_associated_token_account::{
    instruction as ata_ix,
    get_associated_token_address,
    id as ata_program_id,
};

use mpl_token_metadata::ID as MPL_TOKEN_METADATA_ID;

// Bring in your contract's module and types
use mi_smart_contract::{
    id as my_program_id,
    MagaIndexState,
    FractionAccount,
    EnhancedOracleAggregator,
    OracleFeed,
    FeedKind,
    AdaptiveCircuitBreaker,
    MarketState,
    AdvancedGovernance,
    OptimizedVecMap,
    VotingConfig,
    TimelockConfig,
    EmergencyConfig,
    Role,
    VerifiableState,
    DecayRates,
};

////////////////////////////////////////////////////////////////////////////////
// Instruction structs + discriminators for each step
////////////////////////////////////////////////////////////////////////////////

// 1) InitializeFullState
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

fn initialize_full_state_discriminator() -> [u8; 8] {
    let ix_name = b"global:initialize_full_state";
    let hash = anchor_lang::solana_program::hash::hash(ix_name);
    let mut disc = [0u8; 8];
    disc.copy_from_slice(&hash.to_bytes()[..8]);
    disc
}

// 2) MintFractionBatch
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

// 3) PerformUpkeepBatch
#[derive(AnchorSerialize, AnchorDeserialize)]
struct PerformUpkeepBatchIxArgs {
    day: u64,
    max_users: u64,
}

fn perform_upkeep_batch_discriminator() -> [u8; 8] {
    let ix_name = b"global:perform_upkeep_batch";
    let hash = anchor_lang::solana_program::hash::hash(ix_name);
    let mut disc = [0u8; 8];
    disc.copy_from_slice(&hash.to_bytes()[..8]);
    disc
}

// 4) PerformUpkeep
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

////////////////////////////////////////////////////////////////////////////////
// Single end-to-end test
////////////////////////////////////////////////////////////////////////////////

#[tokio::test]
async fn test_e2e_flow_initialize_mint_upkeep() {
    // -------------------------------------------------------------------------
    // A) Create the ProgramTest instance.
    // -------------------------------------------------------------------------
    let program_id = my_program_id();
    let mut program_test = ProgramTest::new(
        "mi_smart_contract",
        program_id,
        processor!(mi_smart_contract::entry),
    );

    // If you have the real mpl_token_metadata.so, add it like so:
    // (Otherwise, you can do a mock add_program with a dummy processor.)
    program_test.add_program(
        "mpl_token_metadata",  // just a label
        MPL_TOKEN_METADATA_ID, // the actual program ID
        None,                  // `None` means "use the built-in BPF from test validator if present"
    );

    // Or if you had a local .so file, you'd do:
    // program_test.add_bpf_program(
    //     "mpl_token_metadata",
    //     MPL_TOKEN_METADATA_ID,
    //     "path/to/mpl_token_metadata.so"
    // );

    // Create keypairs for authority and state account.
    let authority = Keypair::new();
    let maga_index_state = Keypair::new();
    let reward_treasury = Keypair::new();

    // Fund them.
    program_test.add_account(
        authority.pubkey(),
        SolanaAccount {
            lamports: 1_100_000_000_000,
            data: vec![],
            owner: system_program::ID,
            executable: false,
            rent_epoch: 0,
        },
    );
    program_test.add_account(
        reward_treasury.pubkey(),
        SolanaAccount {
            lamports: 1_000_000_000_000,
            data: vec![],
            owner: system_program::ID,
            executable: false,
            rent_epoch: 0,
        },
    );

    let (mut banks_client, _unused_payer, recent_blockhash) = program_test.start().await;

    // -------------------------------------------------------------------------
    // B) Initialize the state.
    // -------------------------------------------------------------------------
    let aggregator_conf = EnhancedOracleAggregator {
        feeds: vec![OracleFeed {
            feed_pubkey: Pubkey::new_unique(),
            previous_price_i64: 0,
            previous_confidence: 0,
            kind: FeedKind::Price,
        }],
        weights: vec![100],
        confidence_thresholds: vec![10_000],
        staleness_thresholds: vec![3600],
        deviation_thresholds: vec![u64::MAX],
        minimum_feeds: 1,
    };

    let circuit_breaker = AdaptiveCircuitBreaker {
        base_threshold: 1000,
        volume_threshold: 1,
        market_state: MarketState::Normal,
    };

    let mut authorities_map = OptimizedVecMap::new();
    authorities_map.insert(authority.pubkey(), Role::Admin).unwrap();

    let gov_conf = AdvancedGovernance {
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
        initial_governor: authority.pubkey(),
    };

    let ver_state = VerifiableState { dummy: 123 };
    let decay_rates = DecayRates {
        winning_side_daily_bps: 50,
        losing_side_daily_bps: 100,
    };

    let init_args = InitializeFullStateIxArgs {
        aggregator_conf,
        circuit_breaker,
        gov: gov_conf,
        ver_state,
        timelock_duration: 0,
        min_valid_price: 1_000_000_000,
        twap_window: 3600,
        max_history_size: 10,
        decay_rates,
    };

    let mut init_data = Vec::new();
    init_data.extend_from_slice(&initialize_full_state_discriminator());
    init_data.extend_from_slice(&init_args.try_to_vec().unwrap());

    let init_ix_accounts = vec![
        AccountMeta::new(maga_index_state.pubkey(), true),
        AccountMeta::new(authority.pubkey(), true),
        AccountMeta::new(reward_treasury.pubkey(), true),
        AccountMeta::new_readonly(system_program::ID, false),
    ];

    let init_ix = solana_program::instruction::Instruction {
        program_id,
        accounts: init_ix_accounts,
        data: init_data,
    };

    let mut init_tx = Transaction::new_with_payer(&[init_ix], Some(&authority.pubkey()));
    init_tx.sign(&[&authority, &maga_index_state, &reward_treasury], recent_blockhash);
    banks_client.process_transaction(init_tx).await.unwrap();

    // -------------------------------------------------------------------------
    // C) Read the clock sysvar and compute the expected day.
    // -------------------------------------------------------------------------
    let clock_after_init = banks_client.get_sysvar::<ProgramClock>().await.unwrap();
    let expected_day = clock_after_init.unix_timestamp / 86400;
    println!("Expected day from clock: {}", expected_day);

    // -------------------------------------------------------------------------
    // D) MintFractionBatch using the computed day.
    // -------------------------------------------------------------------------
    let fraction_account = Keypair::new();
    let deadline = clock_after_init.unix_timestamp + 300;

    let mint_args = MintFractionBatchIxArgs {
        day: expected_day as u64,
        is_bull: true,
        quantity: 100,
        max_price: 2_000_000_000,
        deadline,
    };

    let mut mint_data = Vec::new();
    mint_data.extend_from_slice(&mint_fraction_batch_discriminator());
    mint_data.extend_from_slice(&mint_args.try_to_vec().unwrap());

    let mint_ix_accounts = vec![
        AccountMeta::new(maga_index_state.pubkey(), false),
        AccountMeta::new(fraction_account.pubkey(), true),
        AccountMeta::new(authority.pubkey(), true),
        AccountMeta::new(reward_treasury.pubkey(), true),
        AccountMeta::new_readonly(system_program::ID, false),
    ];

    let mint_ix = solana_program::instruction::Instruction {
        program_id,
        accounts: mint_ix_accounts,
        data: mint_data,
    };

    let bh2 = banks_client.get_latest_blockhash().await.unwrap();
    let mut mint_tx = Transaction::new_with_payer(&[mint_ix], Some(&authority.pubkey()));
    mint_tx.sign(&[&authority, &fraction_account], bh2);
    banks_client.process_transaction(mint_tx).await.unwrap();

    // Verify the fraction account is active.
    let fraction_data = banks_client
        .get_account(fraction_account.pubkey())
        .await
        .unwrap()
        .expect("FractionAccount missing");
    let mut fraction_slice: &[u8] = &fraction_data.data;
    let fraction_parsed = FractionAccount::try_deserialize(&mut fraction_slice).unwrap();
    assert!(fraction_parsed.is_active, "Fraction account not active?");

    // -------------------------------------------------------------------------
    // E) Create an SPL Mint + ATA for the daily NFT.
    // -------------------------------------------------------------------------
    let nft_mint = Keypair::new();
    let mint_rent_exempt = banks_client
        .get_rent()
        .await
        .unwrap()
        .minimum_balance(SplMint::LEN);

    let create_mint_ix = solana_sdk::system_instruction::create_account(
        &authority.pubkey(),
        &nft_mint.pubkey(),
        mint_rent_exempt,
        SplMint::LEN as u64,
        &spl_token_program_id(),
    );

    let init_mint_ix = token_ix::initialize_mint(
        &spl_token_program_id(),
        &nft_mint.pubkey(),
        &authority.pubkey(),
        None,
        0,
    )
    .unwrap();

    let winner_ata = get_associated_token_address(&authority.pubkey(), &nft_mint.pubkey());
    let create_ata_ix = ata_ix::create_associated_token_account(
        &authority.pubkey(),
        &authority.pubkey(),
        &nft_mint.pubkey(),
        &spl_token_program_id(),
    );

    let bh3 = banks_client.get_latest_blockhash().await.unwrap();
    let mut mint_setup_tx = Transaction::new_with_payer(
        &[create_mint_ix, init_mint_ix, create_ata_ix],
        Some(&authority.pubkey()),
    );
    mint_setup_tx.sign(&[&authority, &nft_mint], bh3);
    banks_client.process_transaction(mint_setup_tx).await.unwrap();

    // -------------------------------------------------------------------------
    // F) Call perform_upkeep_batch to start batching/consolidation.
    // -------------------------------------------------------------------------
    let batch_args = PerformUpkeepBatchIxArgs {
        day: expected_day as u64,
        max_users: 100,
    };

    let mut batch_data = Vec::new();
    batch_data.extend_from_slice(&perform_upkeep_batch_discriminator());
    batch_data.extend_from_slice(&batch_args.try_to_vec().unwrap());

    let batch_ix_accounts = vec![
        AccountMeta::new(maga_index_state.pubkey(), false),
        AccountMeta::new(authority.pubkey(), true),
    ];

    let batch_ix = solana_program::instruction::Instruction {
        program_id,
        accounts: batch_ix_accounts,
        data: batch_data,
    };

    let bh_batch = banks_client.get_latest_blockhash().await.unwrap();
    let mut batch_tx = Transaction::new_with_payer(&[batch_ix], Some(&authority.pubkey()));
    batch_tx.sign(&[&authority], bh_batch);
    banks_client.process_transaction(batch_tx).await.unwrap();

    // -------------------------------------------------------------------------
    // G) Now call PerformUpkeep using the same day.
    // -------------------------------------------------------------------------
    let upkeep_args = PerformUpkeepIxArgs {
        day: expected_day as u64
    };

    let mut upkeep_data = Vec::new();
    upkeep_data.extend_from_slice(&perform_upkeep_discriminator());
    upkeep_data.extend_from_slice(&upkeep_args.try_to_vec().unwrap());

    // Derive the mint_authority PDA for the NFT mint authority.
    let (mint_authority_pda, _mint_bump) =
        Pubkey::find_program_address(&[b"MAGA_INDEX_REWARD_POOL", program_id.as_ref()], &program_id);

    // Build the PerformUpkeep instruction accounts in the correct order:
    let upkeep_ix_accounts = vec![
        AccountMeta::new(maga_index_state.pubkey(), false),
        AccountMeta::new(authority.pubkey(), true),
        AccountMeta::new(reward_treasury.pubkey(), true),
        AccountMeta::new(nft_mint.pubkey(), false),
        AccountMeta::new(mint_authority_pda, false),
        AccountMeta::new(winner_ata, false),
        // Use authority again as the "edition_authority" just for this test
        AccountMeta::new(authority.pubkey(), true),
        AccountMeta::new_readonly(MPL_TOKEN_METADATA_ID, false),
        AccountMeta::new_readonly(spl_token_program_id(), false),
        AccountMeta::new_readonly(system_program::ID, false),
        AccountMeta::new_readonly(ata_program_id(), false),
        AccountMeta::new_readonly(clock::ID, false),
        AccountMeta::new_readonly(rent::ID, false),
        AccountMeta::new_readonly(instructions::ID, false),
    ];

    let upkeep_ix = solana_program::instruction::Instruction {
        program_id,
        accounts: upkeep_ix_accounts,
        data: upkeep_data,
    };

    let bh4 = banks_client.get_latest_blockhash().await.unwrap();
    let mut upkeep_tx = Transaction::new_with_payer(&[upkeep_ix], Some(&authority.pubkey()));
    // Sign with authority, reward_treasury, and authority again for edition
    upkeep_tx.sign(&[&authority, &reward_treasury], bh4);
    banks_client.process_transaction(upkeep_tx).await.unwrap();

    // -------------------------------------------------------------------------
    // H) Verifications.
    // -------------------------------------------------------------------------
    let maga_index_state_acc = banks_client
        .get_account(maga_index_state.pubkey())
        .await
        .unwrap()
        .expect("MagaIndexState not found");
    let mut slice: &[u8] = &maga_index_state_acc.data;
    let updated_state = MagaIndexState::try_deserialize(&mut slice)
        .expect("Failed to parse updated MagaIndexState");

    let is_settled = updated_state
        .settled_days
        .get(&(expected_day as u64))
        .cloned()
        .unwrap_or(false);
    assert!(is_settled, "Day was not settled by perform_upkeep!");

    let ata_acc = banks_client
        .get_account(winner_ata)
        .await
        .unwrap()
        .expect("Winner ATA not found");
    let token_account = SplTokenAccount::unpack_from_slice(&ata_acc.data)
        .expect("Failed to parse winner ATA");
    assert_eq!(token_account.amount, 1, "No NFT minted to winner ATA");

    println!("All steps (Initialize, MintFractionBatch, PerformUpkeepBatch, PerformUpkeep) succeeded!");
}
