// # FILE-SIZE-OK: Single-file iced app, splitting would fragment canvas/subscription/state
// Issue #108 — live streaming range bar chart (iced canvas, same drawing stack as flowsurface)
//
// Two modes:
//   Live (default):  mise run sketch -- --symbol BTCUSDT --threshold 250
//   Static CSV:      mise run sketch:csv path/to/bars.csv
//
// Live mode bypasses LiveBarEngine and uses BinanceWebSocketStream + RangeBarProcessor
// directly, giving access to the forming (incomplete) bar for real-time rendering.

use std::collections::VecDeque;

use clap::Parser;
use iced::mouse::Cursor;
use iced::widget::canvas::{self, Cache, Geometry, Path, Stroke};
use iced::widget::Canvas;
use iced::{Color, Element, Fill, Point, Rectangle, Renderer, Size, Subscription, Theme};
use iced::futures::SinkExt;
use rangebar_core::{FixedPoint, RangeBarProcessor, RangeBar};
use rangebar_providers::binance::{BinanceWebSocketStream, ReconnectionPolicy};
use tokio_util::sync::CancellationToken;

/// Subscription input data — must be Hash for iced dedup.
#[derive(Debug, Clone, Hash)]
struct StreamParams {
    symbol: String,
    threshold: u32,
}

fn main() -> iced::Result {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "rangebar_chart=info,rangebar_providers=warn".parse().unwrap()),
        )
        .init();

    let args = Args::parse();

    iced::application(move || boot(args.clone()), App::update, App::view)
        .title("rangebar-chart")
        .window_size(Size::new(1200.0, 600.0))
        .subscription(App::subscription)
        .run()
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser, Debug, Clone)]
#[command(about = "Range bar chart renderer with live Binance streaming")]
struct Args {
    /// Symbol to stream (e.g., BTCUSDT)
    #[arg(short, long, default_value = "BTCUSDT")]
    symbol: String,

    /// Threshold in decimal basis points
    #[arg(short, long, default_value_t = 250)]
    threshold: u32,

    /// Optional CSV file to load historical bars (disables live streaming)
    #[arg(short, long)]
    csv: Option<String>,
}

// ---------------------------------------------------------------------------
// Data
// ---------------------------------------------------------------------------

#[allow(dead_code)] // volume used later for volume sub-chart
#[derive(Debug, Clone)]
struct Bar {
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
}

impl Bar {
    fn from_rangebar(rb: &RangeBar) -> Self {
        Self {
            open: rb.open.to_f64(),
            high: rb.high.to_f64(),
            low: rb.low.to_f64(),
            close: rb.close.to_f64(),
            volume: FixedPoint(rb.volume as i64).to_f64(),
        }
    }
}

/// Try loading OHLCV from a CSV file.
fn load_csv(path: &str) -> Result<Vec<Bar>, Box<dyn std::error::Error>> {
    let mut rdr = csv::ReaderBuilder::new().flexible(true).from_path(path)?;
    let headers = rdr.headers()?.clone();

    let col = |name: &str| headers.iter().position(|h| h == name);
    let i_open = col("open").ok_or("missing 'open' column")?;
    let i_high = col("high").ok_or("missing 'high' column")?;
    let i_low = col("low").ok_or("missing 'low' column")?;
    let i_close = col("close").ok_or("missing 'close' column")?;
    let i_vol = col("volume");

    let mut bars = Vec::new();
    for row in rdr.records() {
        let row = row?;
        let parse = |i: usize| row.get(i).unwrap_or("0").parse::<f64>().unwrap_or(0.0);
        bars.push(Bar {
            open: parse(i_open),
            high: parse(i_high),
            low: parse(i_low),
            close: parse(i_close),
            volume: i_vol.map_or(0.0, |i| parse(i)),
        });
    }
    Ok(bars)
}

fn sample_bars() -> Vec<Bar> {
    let data: &[(f64, f64, f64, f64, f64)] = &[
        (97000.0, 97250.0, 96800.0, 97100.0, 12.5),
        (97100.0, 97400.0, 97050.0, 97350.0, 8.3),
        (97350.0, 97500.0, 97100.0, 97150.0, 15.1),
        (97150.0, 97200.0, 96900.0, 96950.0, 11.7),
        (96950.0, 97100.0, 96750.0, 96800.0, 9.4),
        (96800.0, 97050.0, 96700.0, 97000.0, 13.2),
        (97000.0, 97300.0, 96950.0, 97250.0, 7.8),
        (97250.0, 97500.0, 97200.0, 97450.0, 10.6),
        (97450.0, 97700.0, 97400.0, 97650.0, 14.3),
        (97650.0, 97800.0, 97500.0, 97550.0, 11.9),
        (97550.0, 97600.0, 97300.0, 97350.0, 16.2),
        (97350.0, 97500.0, 97200.0, 97450.0, 8.7),
        (97450.0, 97700.0, 97400.0, 97680.0, 12.1),
        (97680.0, 97900.0, 97650.0, 97850.0, 9.5),
        (97850.0, 98100.0, 97800.0, 98050.0, 13.8),
        (98050.0, 98200.0, 97900.0, 97950.0, 10.2),
        (97950.0, 98000.0, 97700.0, 97750.0, 15.6),
        (97750.0, 97900.0, 97600.0, 97850.0, 11.4),
        (97850.0, 98100.0, 97800.0, 98080.0, 8.9),
        (98080.0, 98300.0, 98050.0, 98250.0, 14.7),
    ];
    data.iter()
        .map(|&(o, h, l, c, v)| Bar { open: o, high: h, low: l, close: c, volume: v })
        .collect()
}

// ---------------------------------------------------------------------------
// Application
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum Message {
    BarCompleted(Bar),
    FormingBarUpdate(Bar),
    TradeCount(u64),
    ConnectionStatus(ConnectionState),
}

#[derive(Debug, Clone)]
enum ConnectionState {
    Connecting,
    Connected,
    Error(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Mode {
    Live,
    Static,
}

fn boot(args: Args) -> (App, iced::Task<Message>) {
    let (bars, mode) = if let Some(ref csv_path) = args.csv {
        match load_csv(csv_path) {
            Ok(bars) => {
                eprintln!("Loaded {} bars from {csv_path}", bars.len());
                (VecDeque::from(bars), Mode::Static)
            }
            Err(e) => {
                eprintln!("Failed to load CSV: {e} — using sample data");
                (VecDeque::from(sample_bars()), Mode::Static)
            }
        }
    } else {
        (VecDeque::new(), Mode::Live)
    };

    let app = App {
        bars,
        forming_bar: None,
        trade_count: 0,
        connection: ConnectionState::Connecting,
        cache: Cache::default(),
        max_visible_bars: 200,
        symbol: args.symbol,
        threshold: args.threshold,
        mode,
    };
    (app, iced::Task::none())
}

struct App {
    bars: VecDeque<Bar>,
    forming_bar: Option<Bar>,
    trade_count: u64,
    connection: ConnectionState,
    cache: Cache,
    max_visible_bars: usize,
    symbol: String,
    threshold: u32,
    mode: Mode,
}

impl App {
    fn update(&mut self, msg: Message) {
        match msg {
            Message::BarCompleted(bar) => {
                self.bars.push_back(bar);
                if self.bars.len() > self.max_visible_bars {
                    self.bars.pop_front();
                }
                self.forming_bar = None;
                self.cache.clear();
            }
            Message::FormingBarUpdate(bar) => {
                self.forming_bar = Some(bar);
                self.cache.clear();
            }
            Message::TradeCount(count) => {
                self.trade_count = count;
                // Don't clear cache for trade count alone — too frequent
            }
            Message::ConnectionStatus(state) => {
                self.connection = state;
                self.cache.clear();
            }
        }
    }

    fn view(&self) -> Element<'_, Message> {
        Canvas::new(self)
            .width(Fill)
            .height(Fill)
            .into()
    }

    fn subscription(&self) -> Subscription<Message> {
        if self.mode == Mode::Static {
            return Subscription::none();
        }

        let params = StreamParams {
            symbol: self.symbol.clone(),
            threshold: self.threshold,
        };

        Subscription::run_with(params, stream_bars)
    }
}

// ---------------------------------------------------------------------------
// Streaming subscription — direct WS + processor (bypasses LiveBarEngine)
// ---------------------------------------------------------------------------

fn stream_bars(
    params: &StreamParams,
) -> std::pin::Pin<Box<dyn iced::futures::Stream<Item = Message> + Send>> {
    let symbol = params.symbol.clone();
    let threshold = params.threshold;

    Box::pin(iced::stream::channel(100, move |mut sender: iced::futures::channel::mpsc::Sender<Message>| async move {
        sender
            .send(Message::ConnectionStatus(ConnectionState::Connecting))
            .await
            .ok();

        // Create processor directly — gives us access to get_incomplete_bar()
        let mut processor = match RangeBarProcessor::new(threshold) {
            Ok(p) => p,
            Err(e) => {
                sender
                    .send(Message::ConnectionStatus(ConnectionState::Error(
                        format!("Processor init failed: {e}"),
                    )))
                    .await
                    .ok();
                std::future::pending::<()>().await;
                return;
            }
        };

        // Trade channel — WS writes, we read
        let (trade_tx, mut trade_rx) = tokio::sync::mpsc::channel(1000);

        // Spawn WS with auto-reconnect in background
        let shutdown = CancellationToken::new();
        let ws_shutdown = shutdown.clone();
        tokio::spawn(async move {
            BinanceWebSocketStream::run_with_reconnect(
                &symbol,
                trade_tx,
                ReconnectionPolicy::default(),
                ws_shutdown,
            )
            .await;
        });

        sender
            .send(Message::ConnectionStatus(ConnectionState::Connected))
            .await
            .ok();

        let mut trade_count: u64 = 0;
        let mut bars_completed: u64 = 0;

        loop {
            match trade_rx.recv().await {
                Some(trade) => {
                    trade_count += 1;

                    if trade_count == 1 {
                        tracing::info!(
                            price = %trade.price.to_f64(),
                            "First trade received"
                        );
                    }

                    match processor.process_single_trade(&trade) {
                        Ok(Some(bar)) => {
                            bars_completed += 1;
                            tracing::info!(
                                bars = bars_completed,
                                trades = trade_count,
                                close = %bar.close.to_f64(),
                                "Bar completed"
                            );
                            sender
                                .send(Message::BarCompleted(Bar::from_rangebar(&bar)))
                                .await
                                .ok();
                        }
                        Ok(None) => {
                            // Send forming bar on every trade for first 50, then every 5th
                            if trade_count <= 50 || trade_count % 5 == 0 {
                                if let Some(incomplete) = processor.get_incomplete_bar() {
                                    sender
                                        .send(Message::FormingBarUpdate(Bar::from_rangebar(&incomplete)))
                                        .await
                                        .ok();
                                }
                            }
                        }
                        Err(e) => {
                            tracing::warn!("Processing error: {e}");
                        }
                    }

                    // Send trade count every 25 trades
                    if trade_count % 25 == 0 {
                        sender
                            .send(Message::TradeCount(trade_count))
                            .await
                            .ok();
                    }
                }
                None => {
                    // WS sender dropped — connection lost
                    sender
                        .send(Message::ConnectionStatus(ConnectionState::Error(
                            "WebSocket disconnected".to_string(),
                        )))
                        .await
                        .ok();
                    break;
                }
            }
        }

        // Keep subscription alive for iced
        std::future::pending::<()>().await;
    }))
}

// ---------------------------------------------------------------------------
// Canvas drawing
// ---------------------------------------------------------------------------

const BULL_COLOR: Color = Color::from_rgb(0.16, 0.71, 0.47);
const BEAR_COLOR: Color = Color::from_rgb(0.90, 0.30, 0.24);
const BG_COLOR: Color = Color::from_rgb(0.08, 0.08, 0.10);
const GRID_COLOR: Color = Color::from_rgba(1.0, 1.0, 1.0, 0.06);
const LABEL_COLOR: Color = Color::from_rgba(1.0, 1.0, 1.0, 0.4);
const FORMING_ALPHA: f32 = 0.4;

impl canvas::Program<Message> for App {
    type State = ();

    fn draw(
        &self,
        _state: &(),
        renderer: &Renderer,
        _theme: &Theme,
        bounds: Rectangle,
        _cursor: Cursor,
    ) -> Vec<Geometry> {
        let geometry = self.cache.draw(renderer, bounds.size(), |frame| {
            let w = bounds.width;
            let h = bounds.height;

            // Background
            frame.fill(&Path::rectangle(Point::ORIGIN, Size::new(w, h)), BG_COLOR);

            // Status bar (top, live mode only)
            if self.mode == Mode::Live {
                let (dot_color, status_text) = match &self.connection {
                    ConnectionState::Connecting => (
                        Color::from_rgb(1.0, 0.85, 0.0),
                        "Connecting...".to_string(),
                    ),
                    ConnectionState::Connected => {
                        let trades_info = if self.trade_count > 0 {
                            format!(" — {} trades", self.trade_count)
                        } else {
                            String::new()
                        };
                        (
                            Color::from_rgb(0.16, 0.71, 0.47),
                            format!("LIVE — {}@{}{}", self.symbol, self.threshold, trades_info),
                        )
                    }
                    ConnectionState::Error(e) => (
                        Color::from_rgb(0.90, 0.30, 0.24),
                        format!("ERROR — {e}"),
                    ),
                };

                // Status dot
                frame.fill(
                    &Path::circle(Point::new(15.0, 10.0), 4.0),
                    dot_color,
                );

                // Status text
                frame.fill_text(canvas::Text {
                    content: status_text,
                    position: Point::new(25.0, 4.0),
                    color: LABEL_COLOR,
                    size: 11.0.into(),
                    ..canvas::Text::default()
                });
            }

            // Collect all bars to draw (completed + forming)
            let total_bars: Vec<(&Bar, bool)> = self
                .bars
                .iter()
                .map(|b| (b, false))
                .chain(self.forming_bar.as_ref().map(|b| (b, true)))
                .collect();

            if total_bars.is_empty() {
                // Show waiting message in live mode
                if self.mode == Mode::Live {
                    let msg = if self.trade_count > 0 {
                        format!("Receiving trades ({})... building first bar", self.trade_count)
                    } else {
                        "Waiting for trades...".to_string()
                    };
                    frame.fill_text(canvas::Text {
                        content: msg,
                        position: Point::new(w / 2.0 - 100.0, h / 2.0),
                        color: LABEL_COLOR,
                        size: 14.0.into(),
                        ..canvas::Text::default()
                    });
                }
                return;
            }

            // Price range across all bars (with minimum padding for single-tick bars)
            let raw_min = total_bars
                .iter()
                .map(|(b, _)| b.low)
                .fold(f64::MAX, f64::min);
            let raw_max = total_bars
                .iter()
                .map(|(b, _)| b.high)
                .fold(f64::MIN, f64::max);
            // Ensure at least 0.1% padding so single-price bars are visible
            let mid = (raw_min + raw_max) / 2.0;
            let min_span = mid * 0.001; // 0.1% of price
            let price_range = (raw_max - raw_min).max(min_span).max(1.0);
            let price_min = mid - price_range / 2.0;
            let price_max = mid + price_range / 2.0;
            let _ = price_max; // used implicitly via price_range

            // Layout constants
            let margin_left = 80.0_f32;
            let margin_right = 20.0_f32;
            let margin_top = 25.0_f32; // extra space for status bar
            let margin_bottom = 40.0_f32;
            let chart_w = w - margin_left - margin_right;
            let chart_h = h - margin_top - margin_bottom;
            let n = total_bars.len() as f32;
            let bar_spacing = (chart_w / n).min(40.0);
            let body_width = (bar_spacing * 0.7).max(3.0);

            let price_to_y = |price: f64| -> f32 {
                let frac = (price - price_min) / price_range;
                margin_top + chart_h * (1.0 - frac as f32)
            };

            // Horizontal grid lines (5 levels)
            for i in 0..=5 {
                let frac = i as f64 / 5.0;
                let price = price_min + frac * price_range;
                let y = price_to_y(price);

                frame.stroke(
                    &Path::line(
                        Point::new(margin_left, y),
                        Point::new(w - margin_right, y),
                    ),
                    Stroke::default().with_width(0.5).with_color(GRID_COLOR),
                );

                frame.fill_text(canvas::Text {
                    content: format!("{:.0}", price),
                    position: Point::new(5.0, y - 6.0),
                    color: LABEL_COLOR,
                    size: 11.0.into(),
                    ..canvas::Text::default()
                });
            }

            // Draw bars
            for (i, (bar, is_forming)) in total_bars.iter().enumerate() {
                let x_center = margin_left + (i as f32 + 0.5) * bar_spacing;
                let bullish = bar.close >= bar.open;
                let base_color = if bullish { BULL_COLOR } else { BEAR_COLOR };

                let color = if *is_forming {
                    Color { a: FORMING_ALPHA, ..base_color }
                } else {
                    base_color
                };

                let y_high = price_to_y(bar.high);
                let y_low = price_to_y(bar.low);
                let y_open = price_to_y(bar.open);
                let y_close = price_to_y(bar.close);

                // Wick
                let wick_width = if *is_forming { 0.5 } else { 1.0 };
                frame.stroke(
                    &Path::line(
                        Point::new(x_center, y_high),
                        Point::new(x_center, y_low),
                    ),
                    Stroke::default().with_width(wick_width).with_color(color),
                );

                // Body
                let body_top = y_open.min(y_close);
                let body_height = (y_open - y_close).abs().max(1.0);
                frame.fill(
                    &Path::rectangle(
                        Point::new(x_center - body_width / 2.0, body_top),
                        Size::new(body_width, body_height),
                    ),
                    color,
                );
            }

            // Title bar
            let mode_label = match &self.mode {
                Mode::Live => match &self.connection {
                    ConnectionState::Connected => "LIVE",
                    ConnectionState::Connecting => "CONNECTING",
                    ConnectionState::Error(_) => "ERROR",
                },
                Mode::Static => "STATIC",
            };

            frame.fill_text(canvas::Text {
                content: format!(
                    "rangebar-chart — {}@{} — {} bars — {}",
                    self.symbol,
                    self.threshold,
                    self.bars.len(),
                    mode_label,
                ),
                position: Point::new(margin_left, h - 15.0),
                color: LABEL_COLOR,
                size: 12.0.into(),
                ..canvas::Text::default()
            });
        });

        vec![geometry]
    }
}
