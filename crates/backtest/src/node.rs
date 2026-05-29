// -------------------------------------------------------------------------------------------------
//  Copyright (C) 2015-2026 Nautech Systems Pty Ltd. All rights reserved.
//  https://nautechsystems.io
//
//  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
//  You may not use this file except in compliance with the License.
//  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
// -------------------------------------------------------------------------------------------------

//! Provides a [`BacktestNode`] that orchestrates catalog-driven backtests.

use std::iter::Peekable;

use ahash::{AHashMap, AHashSet};
use nautilus_core::UnixNanos;
use nautilus_model::{
    data::{
        Bar, Data, HasTsInit, IndexPriceUpdate, InstrumentClose, InstrumentStatus, MarkPriceUpdate,
        OptionGreeks, OrderBookDelta, OrderBookDepth10, QuoteTick, TradeTick,
    },
    enums::{BookType, OtoTriggerMode},
    identifiers::{InstrumentId, Venue},
    instruments::Instrument,
    types::Money,
};
use nautilus_persistence::backend::{catalog::ParquetDataCatalog, session::QueryResult};

use crate::{
    config::{BacktestDataConfig, BacktestRunConfig, NautilusDataType, SimulatedVenueConfig},
    engine::BacktestEngine,
    result::BacktestResult,
};

/// Orchestrates catalog-driven backtests from run configurations.
///
/// `BacktestNode` connects the [`ParquetDataCatalog`] with [`BacktestEngine`] to load
/// historical data and run backtests. Supports both oneshot and streaming modes.
#[derive(Debug)]
#[cfg_attr(
    feature = "python",
    pyo3::pyclass(module = "nautilus_trader.core.nautilus_pyo3.backtest", unsendable)
)]
#[cfg_attr(
    feature = "python",
    pyo3_stub_gen::derive::gen_stub_pyclass(module = "nautilus_trader.backtest")
)]
pub struct BacktestNode {
    configs: Vec<BacktestRunConfig>,
    engines: AHashMap<String, BacktestEngine>,
}

impl BacktestNode {
    /// Creates a new [`BacktestNode`] instance.
    ///
    /// Validates that configs are non-empty and internally consistent:
    /// - All data config instrument venues must have a matching venue config.
    /// - L2/L3 book types require order book data in the data configs.
    /// - Data config time ranges must be valid (start <= end).
    ///
    /// # Errors
    ///
    /// Returns an error if `configs` is empty or validation fails.
    pub fn new(configs: Vec<BacktestRunConfig>) -> anyhow::Result<Self> {
        anyhow::ensure!(!configs.is_empty(), "At least one run config is required");
        validate_configs(&configs)?;
        Ok(Self {
            configs,
            engines: AHashMap::new(),
        })
    }

    /// Returns the run configurations.
    #[must_use]
    pub fn configs(&self) -> &[BacktestRunConfig] {
        &self.configs
    }

    /// Builds backtest engines from the run configurations.
    ///
    /// For each config, creates a [`BacktestEngine`], adds venues, and loads
    /// instruments from the catalog.
    ///
    /// # Errors
    ///
    /// Returns an error if engine creation, venue setup, or instrument loading fails.
    pub fn build(&mut self) -> anyhow::Result<()> {
        for config in &self.configs {
            if self.engines.contains_key(config.id()) {
                continue;
            }

            let engine_config = config.engine().clone();
            let mut engine = BacktestEngine::new(engine_config)?;

            for venue_config in config.venues() {
                let starting_balances: Vec<Money> = venue_config
                    .starting_balances()
                    .iter()
                    .map(|s| s.parse::<Money>())
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|e| anyhow::anyhow!("Invalid starting balance: {e}"))?;

                let default_leverage = venue_config.default_leverage();
                let leverages = venue_config.leverages().cloned().unwrap_or_default();
                let margin_model = venue_config.margin_model().cloned();
                let modules = venue_config
                    .modules()
                    .iter()
                    .cloned()
                    .map(Into::into)
                    .collect();
                let fill_model = venue_config.fill_model().cloned().unwrap_or_default();
                let fee_model = venue_config.fee_model().cloned().unwrap_or_default();
                let latency_model = venue_config.latency_model().cloned().map(Into::into);
                let sim_config = SimulatedVenueConfig::builder()
                    .venue(Venue::from(venue_config.name().as_str()))
                    .oms_type(venue_config.oms_type())
                    .account_type(venue_config.account_type())
                    .book_type(venue_config.book_type())
                    .starting_balances(starting_balances)
                    .maybe_base_currency(venue_config.base_currency())
                    .default_leverage(default_leverage)
                    .leverages(leverages)
                    .maybe_margin_model(margin_model)
                    .modules(modules)
                    .fill_model(fill_model)
                    .fee_model(fee_model)
                    .maybe_latency_model(latency_model)
                    .routing(venue_config.routing())
                    .reject_stop_orders(venue_config.reject_stop_orders())
                    .support_gtd_orders(venue_config.support_gtd_orders())
                    .support_contingent_orders(venue_config.support_contingent_orders())
                    .use_position_ids(venue_config.use_position_ids())
                    .use_random_ids(venue_config.use_random_ids())
                    .use_reduce_only(venue_config.use_reduce_only())
                    .use_market_order_acks(venue_config.use_market_order_acks())
                    .bar_execution(venue_config.bar_execution())
                    .bar_adaptive_high_low_ordering(venue_config.bar_adaptive_high_low_ordering())
                    .trade_execution(venue_config.trade_execution())
                    .liquidity_consumption(venue_config.liquidity_consumption())
                    .allow_cash_borrowing(venue_config.allow_cash_borrowing())
                    .frozen_account(venue_config.frozen_account())
                    .queue_position(venue_config.queue_position())
                    .oto_full_trigger(venue_config.oto_trigger_mode() == OtoTriggerMode::Full)
                    .price_protection_points(venue_config.price_protection_points())
                    .liquidation_enabled(venue_config.liquidation_enabled())
                    .liquidation_trigger_ratio(venue_config.liquidation_trigger_ratio())
                    .liquidation_cancel_open_orders(venue_config.liquidation_cancel_open_orders())
                    .build();
                engine.add_venue(sim_config)?;
            }

            for data_config in config.data() {
                let catalog = create_catalog(data_config)?;
                let instr_ids: Vec<InstrumentId> = data_config.get_instrument_ids()?;
                let filter: Option<Vec<String>> = if instr_ids.is_empty() {
                    None
                } else {
                    Some(instr_ids.iter().map(ToString::to_string).collect())
                };

                let instruments = catalog.query_instruments(filter.as_deref())?;

                if !instr_ids.is_empty() && instruments.is_empty() {
                    let ids: Vec<String> = instr_ids.iter().map(ToString::to_string).collect();
                    anyhow::bail!(
                        "No instruments found in catalog for requested IDs: [{}]",
                        ids.join(", ")
                    );
                }

                for instrument in instruments {
                    engine.add_instrument(&instrument)?;
                }
            }

            for venue_config in config.venues() {
                let Some(settlement_prices) = venue_config.settlement_prices() else {
                    continue;
                };
                let venue = Venue::from(venue_config.name().as_str());

                for (instrument_id, raw_price) in settlement_prices {
                    let price = {
                        let cache = engine.kernel().cache.borrow();
                        let instrument = cache.instrument(instrument_id).ok_or_else(|| {
                            anyhow::anyhow!(
                                "No instrument found for settlement price configuration: {instrument_id}"
                            )
                        })?;
                        instrument.make_price(*raw_price)
                    };
                    engine.set_settlement_price(venue, *instrument_id, price)?;
                }
            }

            self.engines.insert(config.id().to_string(), engine);
        }

        Ok(())
    }

    /// Returns a mutable reference to the engine for the given run config ID.
    #[must_use]
    pub fn get_engine_mut(&mut self, id: &str) -> Option<&mut BacktestEngine> {
        self.engines.get_mut(id)
    }

    /// Returns a reference to the engine for the given run config ID.
    #[must_use]
    pub fn get_engine(&self, id: &str) -> Option<&BacktestEngine> {
        self.engines.get(id)
    }

    /// Returns all created backtest engines.
    #[must_use]
    pub fn get_engines(&self) -> Vec<&BacktestEngine> {
        self.engines.values().collect()
    }

    /// Runs all configured backtests and returns results.
    ///
    /// Automatically calls [`build()`](Self::build) if engines have not been created yet.
    /// For each run config, loads data from the catalog and runs the engine.
    /// Supports both oneshot (`chunk_size = None`) and streaming modes.
    ///
    /// # Errors
    ///
    /// Returns an error if building, data loading, or engine execution fails.
    pub fn run(&mut self) -> anyhow::Result<Vec<BacktestResult>> {
        // Auto-build if not already done
        if self.engines.is_empty() {
            self.build()?;
        }

        let mut results = Vec::new();

        for config in &self.configs {
            let engine = self.engines.get_mut(config.id()).ok_or_else(|| {
                anyhow::anyhow!(
                    "Engine not found for config '{}'. Call build() first.",
                    config.id()
                )
            })?;

            match config.chunk_size() {
                None => run_oneshot(engine, config)?,
                Some(chunk_size) => {
                    anyhow::ensure!(chunk_size > 0, "chunk_size must be > 0");
                    run_streaming(engine, config, chunk_size)?;
                }
            }

            results.push(engine.get_result());

            if config.dispose_on_completion() {
                engine.dispose();
            } else {
                engine.clear_data();
            }
        }

        Ok(results)
    }

    /// Creates a [`ParquetDataCatalog`] from a data config.
    ///
    /// # Errors
    ///
    /// Returns an error if the catalog cannot be created from the URI.
    pub fn load_catalog(config: &BacktestDataConfig) -> anyhow::Result<ParquetDataCatalog> {
        create_catalog(config)
    }

    /// Loads data from the catalog for a specific data config.
    ///
    /// # Errors
    ///
    /// Returns an error if catalog creation or data querying fails.
    pub fn load_data_config(
        config: &BacktestDataConfig,
        start: Option<UnixNanos>,
        end: Option<UnixNanos>,
    ) -> anyhow::Result<Vec<Data>> {
        load_data(config, start, end)
    }

    /// Disposes all engines and releases resources.
    pub fn dispose(&mut self) {
        for engine in self.engines.values_mut() {
            engine.dispose();
        }
        self.engines.clear();
    }
}

fn validate_configs(configs: &[BacktestRunConfig]) -> anyhow::Result<()> {
    // Kernel initialization sets a thread-local MessageBus that can only be
    // initialized once per thread, so multiple engines cannot coexist
    anyhow::ensure!(
        configs.len() <= 1,
        "Only one run config per BacktestNode is supported \
         (kernel MessageBus is a thread-local singleton)"
    );

    let mut seen_ids = AHashSet::new();

    for config in configs {
        anyhow::ensure!(
            seen_ids.insert(config.id()),
            "Duplicate run config ID '{}'",
            config.id()
        );

        let venue_names: Vec<String> = config
            .venues()
            .iter()
            .map(|v| v.name().to_string())
            .collect();

        for data_config in config.data() {
            if let (Some(start), Some(end)) = (data_config.start_time(), data_config.end_time()) {
                anyhow::ensure!(
                    start <= end,
                    "Data config start_time ({start}) must be <= end_time ({end})"
                );
            }

            for instrument_id in data_config.get_instrument_ids()? {
                let venue = instrument_id.venue.to_string();
                anyhow::ensure!(
                    venue_names.contains(&venue),
                    "No venue config found for venue '{venue}' (required by instrument {instrument_id})"
                );
            }
        }

        for venue_config in config.venues() {
            let needs_book_data = matches!(
                venue_config.book_type(),
                BookType::L2_MBP | BookType::L3_MBO
            );

            if needs_book_data {
                let venue_name = venue_config.name().to_string();
                let has_book_data = config.data().iter().any(|dc| {
                    let is_book_type = matches!(
                        dc.data_type(),
                        NautilusDataType::OrderBookDelta | NautilusDataType::OrderBookDepth10
                    );

                    if !is_book_type {
                        return false;
                    }

                    // Unfiltered config (no instrument filter) covers all venues
                    let ids = dc.get_instrument_ids().unwrap_or_default();
                    ids.is_empty() || ids.iter().any(|id| id.venue.to_string() == venue_name)
                });
                anyhow::ensure!(
                    has_book_data,
                    "Venue '{venue_name}' has book_type {:?} but no order book data configured",
                    venue_config.book_type()
                );
            }
        }
    }
    Ok(())
}

fn run_oneshot(engine: &mut BacktestEngine, config: &BacktestRunConfig) -> anyhow::Result<()> {
    for data_config in config.data() {
        let data = load_data(data_config, config.start(), config.end())?;
        if data.is_empty() {
            log::warn!("No data found for config: {:?}", data_config.data_type());
            continue;
        }
        engine.add_data(data, data_config.client_id(), false, false)?;
    }

    engine.sort_data();
    engine.run(
        config.start(),
        config.end(),
        Some(config.id().to_string()),
        false,
    )
}

fn run_streaming(
    engine: &mut BacktestEngine,
    config: &BacktestRunConfig,
    chunk_size: usize,
) -> anyhow::Result<()> {
    let data_configs = config.data();

    if data_configs.len() == 1 {
        // Single config: stream directly from catalog iterator without
        // materializing the full dataset, bounded by chunk_size
        let data_config = &data_configs[0];
        let mut catalog = create_catalog(data_config)?;
        let result = dispatch_query(&mut catalog, data_config, config.start(), config.end())?;
        stream_chunks(engine, config, result.peekable(), chunk_size)?;
    } else if can_share_catalog(data_configs) {
        // Multiple configs sharing the same catalog backend: register each
        // data type's query into one DataBackendSession, then take one
        // merged QueryResult (a KMerge over all DataFusion batch streams)
        // and chunk-stream it. Memory stays bounded; we do not materialize
        // per-type Vec<Data> the way the eager fallback below does.
        //
        // We use `stream_chunks_resort` rather than `stream_chunks` because
        // the heterogeneous-data-type KMerge can yield items slightly out
        // of order at batch boundaries — per-stream batches are individually
        // sorted by DataFusion ORDER BY, but cross-batch arrival order in
        // the async EagerStream pipeline is not strictly monotonic when
        // batches for one data type race against batches for another.
        // Re-sorting per chunk is O(chunk_size log chunk_size) which is
        // negligible vs the cost of the engine processing the chunk.
        let mut catalog = create_catalog(&data_configs[0])?;
        catalog.reset_session();
        for data_config in data_configs {
            dispatch_register_query(
                &mut catalog,
                data_config,
                config.start(),
                config.end(),
            )?;
        }
        let merged = catalog.session.get_query_result();
        // Gap #4 from the KMerge OOO investigation: wrap the merged
        // KMerge output in an OrderingVerifier that logs every backward
        // yield with both the offending ts and the immediately-prior
        // yield's ts. Lets us distinguish "KMerge itself is yielding
        // backward" from "yields are monotonic but stream_chunks_resort
        // is computing the wrong chunk boundary".
        let verified = OrderingVerifier::new(merged, "multi-config-kmerge");
        stream_chunks_resort(engine, config, verified.peekable(), chunk_size)?;
    } else {
        // Distinct catalog backends across configs (different protocol /
        // storage options): DataFusion registers object stores per-session
        // so we can't merge into one session. Fall back to eager merge.
        let all_data = load_and_merge_data(config)?;
        stream_chunks(engine, config, all_data.into_iter().peekable(), chunk_size)?;
    }

    Ok(())
}

fn can_share_catalog(configs: &[BacktestDataConfig]) -> bool {
    let first = &configs[0];
    configs.iter().all(|c| {
        c.catalog_path() == first.catalog_path()
            && c.catalog_fs_protocol() == first.catalog_fs_protocol()
            && c.catalog_fs_storage_options() == first.catalog_fs_storage_options()
            && c.catalog_fs_rust_storage_options() == first.catalog_fs_rust_storage_options()
    })
}

// Variant of `dispatch_query` that registers files into the catalog's
// shared session without consuming a QueryResult. Used by the multi-config
// streaming path: callers register N times then call
// `catalog.session.get_query_result()` once to obtain a merged stream.
fn dispatch_register_query(
    catalog: &mut ParquetDataCatalog,
    config: &BacktestDataConfig,
    run_start: Option<UnixNanos>,
    run_end: Option<UnixNanos>,
) -> anyhow::Result<()> {
    let identifiers = config.query_identifiers();
    let start = max_opt(config.start_time(), run_start);
    let end = min_opt(config.end_time(), run_end);
    let filter = config.filter_expr();
    // Force directory-based registration in the multi-config streaming path:
    // file-based registration creates one DataFusion table per parquet file and
    // relies on `file_sort_order` to skip the SortExec node. With many small
    // tables fed into a KMerge across heterogeneous data types, we have seen
    // out-of-order yields that violate the engine's `start <= end` invariant
    // in stream_chunks. Directory-based registration produces one table per
    // (data_type, instrument) directory, which DataFusion handles with a
    // deterministic ordered scan plan across the contained files.
    let optimize = true;
    let _ = config.optimize_file_loading();

    match config.data_type() {
        NautilusDataType::QuoteTick => {
            catalog.register_query::<QuoteTick>(identifiers, start, end, filter, None, optimize)
        }
        NautilusDataType::TradeTick => {
            catalog.register_query::<TradeTick>(identifiers, start, end, filter, None, optimize)
        }
        NautilusDataType::Bar => {
            catalog.register_query::<Bar>(identifiers, start, end, filter, None, optimize)
        }
        NautilusDataType::OrderBookDelta => catalog
            .register_query::<OrderBookDelta>(identifiers, start, end, filter, None, optimize),
        NautilusDataType::OrderBookDepth10 => catalog
            .register_query::<OrderBookDepth10>(identifiers, start, end, filter, None, optimize),
        NautilusDataType::MarkPriceUpdate => catalog
            .register_query::<MarkPriceUpdate>(identifiers, start, end, filter, None, optimize),
        NautilusDataType::IndexPriceUpdate => catalog
            .register_query::<IndexPriceUpdate>(identifiers, start, end, filter, None, optimize),
        NautilusDataType::InstrumentStatus => catalog
            .register_query::<InstrumentStatus>(identifiers, start, end, filter, None, optimize),
        NautilusDataType::InstrumentClose => catalog
            .register_query::<InstrumentClose>(identifiers, start, end, filter, None, optimize),
        NautilusDataType::OptionGreeks => catalog
            .register_query::<OptionGreeks>(identifiers, start, end, filter, None, optimize),
    }
}

// Wraps an `Iterator<Item = Data>` and logs a WARN every time it yields
// an item with `ts_init` strictly less than the previous yield. Diagnostic
// scaffold for the KMerge OOO investigation — answers "is the upstream
// KMerge yielding backward, or is the chunk-boundary math wrong?" without
// requiring a special build. Always-on; cost is one comparison per yield.
struct OrderingVerifier<I: Iterator<Item = Data>> {
    inner: I,
    last_ts: Option<UnixNanos>,
    label: &'static str,
    violations: usize,
}

impl<I: Iterator<Item = Data>> OrderingVerifier<I> {
    fn new(inner: I, label: &'static str) -> Self {
        Self {
            inner,
            last_ts: None,
            label,
            violations: 0,
        }
    }
}

impl<I: Iterator<Item = Data>> Iterator for OrderingVerifier<I> {
    type Item = Data;

    fn next(&mut self) -> Option<Self::Item> {
        let item = self.inner.next()?;
        let ts = item.ts_init();
        if let Some(prev) = self.last_ts
            && ts < prev
        {
            self.violations += 1;
            // Log up to 50 violations in detail (enough to characterise
            // the pattern), then count silently. Avoids drowning the log
            // on pathological inputs.
            if self.violations <= 50 {
                log::warn!(
                    "OrderingVerifier[{}] backward yield #{}: ts={} prev={} gap={}ns",
                    self.label,
                    self.violations,
                    ts,
                    prev,
                    u64::from(prev).saturating_sub(u64::from(ts)),
                );
            } else if self.violations.is_power_of_two() {
                log::warn!(
                    "OrderingVerifier[{}] backward yield count now {} (further violations suppressed unless power-of-two)",
                    self.label,
                    self.violations,
                );
            }
        }
        // Update last_ts to the max seen so a single backward yield
        // doesn't reset the baseline and hide subsequent backward yields.
        self.last_ts = Some(match self.last_ts {
            Some(prev) => prev.max(ts),
            None => ts,
        });
        Some(item)
    }
}

// Multi-config variant of `stream_chunks` that re-sorts each chunk by
// `ts_init` before computing the boundary end timestamp. Required for the
// streaming multi-data-config path where the upstream KMerge may yield
// items slightly out of order across heterogeneous DataFusion batch
// streams.
fn stream_chunks_resort<I: Iterator<Item = Data>>(
    engine: &mut BacktestEngine,
    config: &BacktestRunConfig,
    mut iter: Peekable<I>,
    chunk_size: usize,
) -> anyhow::Result<()> {
    if iter.peek().is_none() {
        engine.end();
        return Ok(());
    }

    let mut next_start = config.start();
    let mut chunk_idx: usize = 0;
    let mut last_yielded_ts: Option<UnixNanos> = None;

    loop {
        let mut chunk = take_aligned_chunk(&mut iter, chunk_size);
        if chunk.is_empty() {
            break;
        }

        // Diagnostic: verify the upstream merged stream is globally
        // ascending by ts_init across chunks. If it is not, the
        // multi-config KMerge has a correctness bug we have not yet
        // identified — log loudly with both prior boundary and chunk min.
        let chunk_min_pre_sort = chunk
            .first()
            .map(HasTsInit::ts_init)
            .expect("non-empty chunk has a first item");
        if let Some(prev) = last_yielded_ts
            && chunk_min_pre_sort < prev
        {
            log::warn!(
                "stream_chunks_resort: chunk {} first ts {} < previous chunk last ts {} \
                 (upstream merge produced backwards-going items; sorting locally to repair)",
                chunk_idx,
                chunk_min_pre_sort,
                prev,
            );
        }

        chunk.sort_by_key(HasTsInit::ts_init);
        let chunk_max = chunk
            .last()
            .map(HasTsInit::ts_init)
            .expect("non-empty chunk has a last item");

        // Even after local sort, the cross-chunk invariant
        // (chunk_max_N+1 >= chunk_max_N) may be violated if items leaked
        // into chunk N+1 with ts strictly less than chunk_max_N. In that
        // case clamp the engine start to chunk_min so the engine can still
        // process this chunk, log a warning, and skip the boundary update.
        let effective_start = if let Some(prev) = last_yielded_ts
            && chunk_max < prev
        {
            log::warn!(
                "stream_chunks_resort: chunk {} max ts {} < previous chunk max ts {}; \
                 clamping engine start to chunk min {} to avoid start>end",
                chunk_idx,
                chunk_max,
                prev,
                chunk_min_pre_sort,
            );
            Some(chunk_min_pre_sort)
        } else {
            next_start
        };

        let is_last = iter.peek().is_none();
        let end = if is_last {
            config.end().or(Some(chunk_max))
        } else {
            Some(chunk_max)
        };

        engine.add_data(chunk, None, false, true)?;
        engine.run(
            effective_start,
            end,
            Some(config.id().to_string()),
            true,
        )?;
        engine.clear_data();

        if engine.kernel().is_shutdown_requested() {
            return Ok(());
        }

        // Only advance the boundary forward — never let a backward chunk
        // pull next_start into the past.
        let new_boundary = match (last_yielded_ts, end) {
            (Some(prev), Some(e)) => Some(prev.max(e)),
            (None, end) => end,
            (some, None) => some,
        };
        next_start = new_boundary;
        last_yielded_ts = new_boundary;
        chunk_idx += 1;
    }

    engine.end();
    Ok(())
}

// Feeds data from an iterator to the engine in timestamp-aligned chunks.
// Each chunk contains up to `chunk_size` events, extended to include all
// events sharing the boundary timestamp so timers flush correctly.
fn stream_chunks<I: Iterator<Item = Data>>(
    engine: &mut BacktestEngine,
    config: &BacktestRunConfig,
    mut iter: Peekable<I>,
    chunk_size: usize,
) -> anyhow::Result<()> {
    if iter.peek().is_none() {
        engine.end();
        return Ok(());
    }

    let mut next_start = config.start();

    loop {
        let chunk = take_aligned_chunk(&mut iter, chunk_size);
        if chunk.is_empty() {
            break;
        }

        let is_last = iter.peek().is_none();
        let end = if is_last {
            config.end()
        } else {
            chunk.last().map(HasTsInit::ts_init)
        };

        engine.add_data(chunk, None, false, true)?;
        engine.run(next_start, end, Some(config.id().to_string()), true)?;
        engine.clear_data();

        // A shutdown request during the chunk already triggered end() inside
        // engine.run(); stop loading further chunks so later data is not processed
        if engine.kernel().is_shutdown_requested() {
            return Ok(());
        }

        // Carry forward the end timestamp so the next chunk's run_impl
        // sets clocks contiguously and processes gap timers correctly
        next_start = end;
    }

    engine.end();
    Ok(())
}

// Takes up to `chunk_size` items, then extends to include all remaining
// items sharing the boundary timestamp to avoid splitting same-ts events.
fn take_aligned_chunk<I: Iterator<Item = Data>>(
    iter: &mut Peekable<I>,
    chunk_size: usize,
) -> Vec<Data> {
    let mut chunk = Vec::with_capacity(chunk_size);

    for _ in 0..chunk_size {
        match iter.next() {
            Some(item) => chunk.push(item),
            None => return chunk,
        }
    }

    if let Some(boundary_ts) = chunk.last().map(HasTsInit::ts_init) {
        while iter.peek().is_some_and(|d| d.ts_init() == boundary_ts) {
            chunk.push(iter.next().unwrap());
        }
    }

    chunk
}

fn load_and_merge_data(config: &BacktestRunConfig) -> anyhow::Result<Vec<Data>> {
    let mut all_data = Vec::new();

    for data_config in config.data() {
        let data = load_data(data_config, config.start(), config.end())?;
        if data.is_empty() {
            log::warn!("No data found for config: {:?}", data_config.data_type());
            continue;
        }
        all_data.extend(data);
    }
    all_data.sort_by_key(HasTsInit::ts_init);
    Ok(all_data)
}

fn create_catalog(config: &BacktestDataConfig) -> anyhow::Result<ParquetDataCatalog> {
    let uri = match config.catalog_fs_protocol() {
        Some(protocol) => format!("{protocol}://{}", config.catalog_path()),
        None => config.catalog_path().to_string(),
    };
    let storage_options = config
        .catalog_fs_rust_storage_options()
        .cloned()
        .or_else(|| config.catalog_fs_storage_options().cloned());
    ParquetDataCatalog::from_uri(&uri, storage_options, None, None, None)
}

fn load_data(
    config: &BacktestDataConfig,
    run_start: Option<UnixNanos>,
    run_end: Option<UnixNanos>,
) -> anyhow::Result<Vec<Data>> {
    let mut catalog = create_catalog(config)?;
    let result = dispatch_query(&mut catalog, config, run_start, run_end)?;
    Ok(result.collect())
}

fn dispatch_query(
    catalog: &mut ParquetDataCatalog,
    config: &BacktestDataConfig,
    run_start: Option<UnixNanos>,
    run_end: Option<UnixNanos>,
) -> anyhow::Result<QueryResult> {
    catalog.reset_session();

    let identifiers = config.query_identifiers();
    let start = max_opt(config.start_time(), run_start);
    let end = min_opt(config.end_time(), run_end);
    let filter = config.filter_expr();
    let optimize = config.optimize_file_loading();

    match config.data_type() {
        NautilusDataType::QuoteTick => {
            catalog.query::<QuoteTick>(identifiers, start, end, filter, None, optimize)
        }
        NautilusDataType::TradeTick => {
            catalog.query::<TradeTick>(identifiers, start, end, filter, None, optimize)
        }
        NautilusDataType::Bar => {
            catalog.query::<Bar>(identifiers, start, end, filter, None, optimize)
        }
        NautilusDataType::OrderBookDelta => {
            catalog.query::<OrderBookDelta>(identifiers, start, end, filter, None, optimize)
        }
        NautilusDataType::OrderBookDepth10 => {
            catalog.query::<OrderBookDepth10>(identifiers, start, end, filter, None, optimize)
        }
        NautilusDataType::MarkPriceUpdate => {
            catalog.query::<MarkPriceUpdate>(identifiers, start, end, filter, None, optimize)
        }
        NautilusDataType::IndexPriceUpdate => {
            catalog.query::<IndexPriceUpdate>(identifiers, start, end, filter, None, optimize)
        }
        NautilusDataType::InstrumentStatus => {
            catalog.query::<InstrumentStatus>(identifiers, start, end, filter, None, optimize)
        }
        NautilusDataType::OptionGreeks => {
            catalog.query::<OptionGreeks>(identifiers, start, end, filter, None, optimize)
        }
        NautilusDataType::InstrumentClose => {
            catalog.query::<InstrumentClose>(identifiers, start, end, filter, None, optimize)
        }
    }
}

fn max_opt(a: Option<UnixNanos>, b: Option<UnixNanos>) -> Option<UnixNanos> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.max(b)),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

fn min_opt(a: Option<UnixNanos>, b: Option<UnixNanos>) -> Option<UnixNanos> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}
