import streamlit as st
import yfinance as yf
import altair as alt # Visuals
import pandas as pd
import numpy as np
import time

import datetime
from deep_translator import GoogleTranslator

# --- TRANSLATION HELPER ---
@st.cache_data(ttl=86400, show_spinner=False)
def translate_text(text, target_lang='th'):
    try:
        if not text: return ""
        # Chunking might be needed for very long text, but summaries are usually < 5000 chars
        translator = GoogleTranslator(source='auto', target=target_lang)
        return translator.translate(text)
    except Exception as e:
        return text # Fallback to original


# --- CACHING HELPERS (Optimization) ---
@st.cache_data(ttl=3600*12, show_spinner=False)
def fetch_cached_info(ticker):
    """Cache the heavy API call for stock metadata (with Retry)."""
    retries = 3
    for attempt in range(retries):
        try:
            return yf.Ticker(ticker).info
        except Exception as e:
            err_msg = str(e).lower()
            if "too many requests" in err_msg or "rate limited" in err_msg or "429" in err_msg:
                if attempt < retries - 1:
                    sleep_time = (2 ** attempt) + (0.1 * (attempt+1)) # Exponential Backoff: 1.1s, 2.2s, 4.3s
                    print(f"[{ticker}] Rate Limited. Retrying in {sleep_time:.2f}s...")
                    time.sleep(sleep_time)
                    continue
            
            print(f"[{ticker}] Info Error: {e}")
            return None # Return None on error for easier fallback trigger
    return None

# Retry Helper for Object access (when we have obj but need property)
def safe_get_info(stock_obj):
    val = None
    try:
        val = stock_obj.info
    except Exception:
        # Retry logic 
        try:
             time.sleep(1)
             val = stock_obj.info
        except:
             pass
    
    return val if val is not None else {}

def get_grade(score):
    if score >= 80: return "A+"
    if score >= 70: return "A"
    if score >= 60: return "B"
    if score >= 50: return "C"
    if score >= 40: return "D"
    return "F"

# ---------------------------------------------------------




@st.cache_data(ttl=3600*12, show_spinner=False)
def fetch_cached_history(ticker, period='5y'):
    """Cache the history fetch for deep analysis (with Retry)."""
    retries = 3
    for attempt in range(retries):
        try:
            return yf.Ticker(ticker).history(period=period)
        except Exception as e:
            err_msg = str(e).lower()
            if "too many requests" in err_msg or "rate limited" in err_msg or "429" in err_msg:
                 if attempt < retries - 1:
                    time.sleep((2 ** attempt))
                    continue
            return pd.DataFrame()
    return pd.DataFrame()

# --- PROFESSIONAL UI OVERHAUL ---
def inject_custom_css():
    st.markdown("""
        <style>
        /* Main Font */
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
        html, body, [class*="css"] {
            font-family: 'Roboto', sans-serif;
        }
        
        /* Custom Keyframes for Page Transitions */
        @keyframes fadeInSlideUp {
            0% { opacity: 0; transform: translateY(20px); filter: blur(5px); }
            100% { opacity: 1; transform: translateY(0); filter: blur(0); }
        }

        @keyframes pulseGlow {
            0% { box-shadow: 0 0 5px rgba(212, 175, 55, 0.2); }
            50% { box-shadow: 0 0 15px rgba(212, 175, 55, 0.5); }
            100% { box-shadow: 0 0 5px rgba(212, 175, 55, 0.2); }
        }

        /* Apply Page Transition to the main content area */
        .block-container {
            padding-top: 1rem;
            animation: fadeInSlideUp 0.8s cubic-bezier(0.16, 1, 0.3, 1) forwards;
            max_width: 1200px;
            padding-left: 2rem;
            padding-right: 2rem;
            margin: auto;
        }

        /* Responsive Breakpoint for Large Screens to prevent stretching */
        @media (min-width: 1200px) {
            .block-container {
                max-width: 1200px !important;
            }
        }
        
        /* Hide Streamlit Header/Toolbar */
        header {visibility: hidden;}
        [data-testid="stToolbar"] {visibility: hidden;}
        .stDeployButton {display:none;}

        /* CFA-Style Blue Header for Tabs (Full Width) */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0px; /* Remove gap between tabs */
            background-color: transparent; 
            padding: 0px;
            border-bottom: 2px solid #D4AF37;
        }

        .stTabs [data-baseweb="tab"] {
            flex-grow: 1; /* Stretch to fill width */
            height: 50px;
            white-space: pre-wrap;
            background-color: #f8f9fa; /* Light gray for unselected */
            transition: all 0.3s ease;
            border-radius: 0px; /* No corners */
            color: #D4AF37; 
            font-weight: 600;
            border: none; /* Clean Look */
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .stTabs [data-baseweb="tab"]:hover {
            background-color: #e9ecef;
            color: #B8860B;
        }

        .stTabs [aria-selected="true"] {
            background-color: #D4AF37 !important; /* Active Gold */
            color: #ffffff !important;
            font-weight: 700;
            transform: scale(1.02);
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        }
        
        /* Metrics & Buttons */
        div[data-testid="stMetricValue"] {
            font-size: 1.4rem !important;
            color: #D4AF37;
            animation: fadeInSlideUp 1s ease-out;
        }
        
        /* Primary Button Gold */
        div.stButton > button:first-child {
            background-color: #D4AF37;
            color: white;
            border-radius: 8px;
            border: none;
            padding: 0.6rem 1.2rem;
            font-weight: 600;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        div.stButton > button:first-child:hover {
            background-color: #B8860B;
            color: white;
            transform: translateY(-2px);
            box-shadow: 0 6px 15px rgba(212, 175, 55, 0.3);
            animation: pulseGlow 2s infinite;
        }
        div.stButton > button:first-child:active {
            transform: translateY(0);
        }

        </style>
    """, unsafe_allow_html=True)

# --- LOCALIZATION & TEXT ASSETS ---

TRANS = {
    'EN': {
        'sidebar_title': "🏛️ Scanner Controls",
        'market_label': "Crypto Category",
        'strategy_label': "Strategy Preset",
        'mode_header': "3. Mode",
        'strict_label': "Select Strictly Enforced Metrics",
        'perf_label': "Performance Metrics",
        'val_header': " On-Chain (Valuation)",
        'prof_header': "⚡ Momentum (Technical)",
        'risk_header': "🛡️ Risk & Volatility",
        'sector_label': "Select Narrative (Optional)",
        'lynch_label': "Select Cycle Phase (Optional)",
        'execute_btn': "🚀 Execute Bidnow Scan",
        'main_title': "Bidnow",
        'scan_limit': "Scan Limit",
        'results_header': "🏆 Top Coins (Cycle & On-Chain Analysis)",
        'stage1_msg': "📡 Stage 1: Fetching Universe...",
        'stage2_msg': "✅ Stage 1 Complete. Analyzing Top Candidates...",
        'no_data': "❌ No coins matched your STRICT criteria.",
        'deep_dive_title': "🔍 Deep Dive Analysis",
        'glossary_title': "📚 Crypto Glossary",
        'howto_title': "📘 How to Use",
        'search_ticker': "Enter Coin Symbol (e.g. BTC-USD, ETH-USD)",
        'analyze_btn': "Analyze Coin",
        'about_title': "ℹ️ About Bidnow",
        'about_desc': "Professional Crypto Analysis Platform using Cycle Theory, On-Chain Metrics (MVRV), and Power Law support bands. Designed for serious investors to find high-probability setups.",
        
        'scanner_config': "🛠️ Scanner Configuration & Settings",
        'univ_scale': "1. Universe & Scale",
        'strat_mandate': "2. Strategy Mandate",
        'crit_thresh': "3. Criteria Thresholds",
        'opt_filters': "Optional Filters",
        'analyze_top_n': "Analyze Top N Deeply (Stage 2)",
        
        # New Glossary Terms (EN) - Professional Curriculum Style
        'gloss_mvrv': "**MVRV Z-Score (Market Value to Realized Value)**: An institutional-grade metric used to identify market extremes. "
                      "It calculates the standard deviation of market capitalization compared to the 'realized' capitalization (the aggregate cost basis of all holders).\n\n"
                      "**Methodology**: A Z-Score above 3.5 statistically indicates an 'Overvalued' state (Bubble Zone), while a value below 0.0 suggests an 'Undervalued' state "
                      "(Accumulation Zone). This disparity identifies periods where the unrealized profit/loss of the network is at a historical outlier.",
        'gloss_rsi': "**RSI (Relative Strength Index)**: A technical momentum oscillator that measures the magnitude of recent price changes to evaluate overbought or oversold conditions.\n\n"
                     "**Application**: Values are bound between 0 and 100. Traditionally, an RSI above 70 indicates a security is becoming overbought or overvalued and may be primed for a trend reversal "
                      "or corrective pullback. Conversely, an RSI below 30 indicates an oversold or undervalued condition.",
        'gloss_sharpe': "**Sharpe Ratio**: A mathematical measure of the 'Risk-Adjusted Return' of an asset or portfolio. It is defined as the difference between the returns of the investment "
                        "and the risk-free rate, divided by the standard deviation of its excess returns.\n\n"
                        "**Interpretation**: A ratio above 1.0 is considered acceptable to good. Higher values indicate that the excess return is a result of superior asset selection "
                        "rather than excessive volatility exposure.",
        'gloss_Bidnow_score': "**Bidnow Score (Institutional Grade)**: A multi-factor quantitative model (0-100) assessing the fundamental integrity of a digital asset based on four core pillars:\n\n"
                               "1. **Capital Adequacy & Financial Health (30%)**: Evaluation of protocol revenue generation (DeFiLlama data) and relative valuation (P/S Ratio).\n"
                               "2. **Network Dynamics & Adoption (30%)**: Quantitative analysis of on-chain activity, comparing 7-day average volume against the 30-day baseline to detect usage acceleration.\n"
                               "3. **Technological Infrastructure (20%)**: Assessment of ecosystem maturity, developer activity, and blue-chip classification (L1/L2 status).\n"
                               "4. **Supply-Side Dynamics (20%)**: Analysis of tokenomics, inflation schedules, and circulating supply ratios to mitigate unlock-driven dilution risks.",
        'gloss_cycle': "**Cycle Risk Assessment**: A proprietary gauge determining the asset's proximity to its historical market cycle peak or trough.\n\n"
                       "**Quantitative Range**: Levels below 20% represent the 'Maximum Opportunity' zone (Post-Drawdown), whereas levels above 80% indicate 'Maximum Risk' or 'Euphoria' "
                       "correlated with significant market corrections.",
        
        # --- Asset Categories (EN) ---
        'cat_l1_title': "Layer 1 (L1) - The Sovereign Foundations",
        'cat_l1_desc': "**Layer 1 (L1)** protocols are the sovereign infrastructures of the blockchain world. They operate their own independent ledgers and consensus mechanisms (Proof of Work or Proof of Stake).\n\n"
                       "**Deep Dive Analysis**:\n"
                       "- **The Scalability Trilemma**: Every L1 must balance Security, Decentralization, and Scalability. Bitcoin prioritizes security; Solana prioritizes speed.\n"
                       "- **Economic Value**: L1 tokens are 'Digital Real Estate'. Value is driven by network effects—the more developers and applications (dApps) built on top, the higher the demand for the native gas token.\n"
                       "- **Consensus Dynamics**: POX (Proof of Stake) models allow for 'Yield' via staking, creating a floor price for long-term holders.\n"
                       "- **Risk Factors**: High competition ('L1 Wars'). If developers migrate to a faster/cheaper chain, the network can lose its premium valuation rapidly.\n"
                       "- **Primary Examples**: Bitcoin (Digital Gold), Ethereum (The World Computer), Solana (High-Frequency Infrastructure).",
        
        'cat_l2_title': "Layer 2 (L2) - Scaling the Frontier",
        'cat_l2_desc': "**Layer 2 (L2)** consists of protocols built on top of an existing L1 (mostly Ethereum) to increase throughput without compromising the security of the underlying base layer.\n\n"
                       "**Deep Dive Analysis**:\n"
                       "- **Rollup Technology**: Transactions are processed off-chain, compressed, and 'rolled up' into a single proof submitted to the L1. Breakdown:\n"
                         "  - *Optimistic Rollups*: Assume transactions are valid unless challenged (e.g., Arbitrum, Optimism).\n"
                         "  - *Zero-Knowledge (ZK) Rollups*: Use complex mathematics (validity proofs) to prove transactions are correct instantly (e.g., ZK-Sync, Starknet).\n"
                       "- **Data Availability**: L2s rely on the L1 for 'Truth'. If the L2 goes down, your funds are still safe on the L1 through 'escape hatches'.\n"
                       "- **The Revenue Model**: L2s keep the difference between the gas fees they collect from users and the rent they pay to the L1.\n"
                       "- **Primary Examples**: Arbitrum, Optimism, Base, Polygon (AggLayer).",
        
        'cat_defi_title': "DeFi - The Global Permissionless Bank",
        'cat_defi_desc': "**Decentralized Finance (DeFi)** replaces traditional intermediaries (banks, brokers) with open-source 'Smart Contracts'.\n\n"
                         "**Deep Dive Analysis**:\n"
                         "- **The AMM Model**: Automated Market Makers like Uniswap replaced 'Order Books' with 'Liquidity Pools'. Prices are determined by the ratio of assets in a pool, allowing 24/7 permissionless swapping.\n"
                         "- **Composability (Money Legos)**: Protocols can be stacked. You can deposit collateral in Aave, take a loan, and provide liquidity in Curve—all in one transaction sequence.\n"
                         "- **Yield Generation**: Driven by protocol fees or 'Liquidity Mining' (rewarding users with tokens). Institutional interest is focused on 'Real Yield' (revenue-based) rather than inflationary rewards.\n"
                         "- **Critical Risks**: Smart Contract exploits, 'Rug Pulls', and Oracle Failures (incorrect price data triggering liquidations).\n"
                         "- **Primary Examples**: Uniswap (Exchange), Aave (Lending), MakerDAO (CDPs/Stablecoins).",
        
        'cat_gamefi_title': "GameFi & The Metaverse Economy",
        'cat_gamefi_desc': "**GameFi** merges Decentralized Finance with the Gaming industry through 'Play-to-Earn' (P2E) models and true digital ownership.\n\n"
                           "**Deep Dive Analysis**:\n"
                           "- **Asset Tokenization**: Using NFTs (ERC-721/1155), in-game items like swords or land become liquid assets that can be traded on open markets (Opensea).\n"
                           "- **The Virtual Economy**: Entire GDPs are formed within virtual worlds. Land ownership allows owners to monetize digital experiences (concerts, advertising).\n"
                           "- **The Sustainability Issue**: Early models (like Axie) faced high inflation. The next generation focuses on 'Play-and-Earn', where fun precedes financial rewards.\n"
                           "- **Market Dynamics**: Extremely high beta (high volatility). These assets often lag the broader market but rally explosively during 'Euphoria' phases.\n"
                           "- **Primary Examples**: Axie Infinity, The Sandbox, Illuvium, Gala Games.",
        
        'cat_meme_title': "Memecoins - Social Consensus Assets",
        'cat_meme_desc': "**Memecoins** are assets backed by social attention, community culture, and internet memes rather than cash flows or technical utility.\n\n"
                         "**Deep Dive Analysis**:\n"
                         "- **Social Consensus**: If millions of people agree a 'dog picture' has value, it has value. This is the ultimate expression of the 'Attention Economy'.\n"
                         "- **Zero-to-One Launch**: Unlike VC-backed projects, many memes launch fairly (no pre-sale), creating a 'cult' community with high loyalty.\n"
                         "- **The Slot Machine Effect**: High speculation attracts retail investors looking for '1000x' gains. This creates a reflexive cycle: Price goes up -> more attention -> more buyers -> price goes up.\n"
                         "- **Risk Assessment**: Extremely high risk of 'Total Loss'. Liquidity is often thin, meaning prices can drop 90% in hours if the trend shifts.\n"
                         "- **Primary Examples**: Dogecoin, Shiba Inu, Pepe, Dogwifhat.",
        
        'cat_ai_title': "AI Protocols - The Distributed Intelligence",
        'cat_ai_desc': "**AI Protocols** build decentralized marketplaces for the components required for Artificial Intelligence: Compute, Data, and Models.\n\n"
                       "**Deep Dive Analysis**:\n"
                       "- **Decentralized Compute**: Instead of relying on Nvidia/Azure, protocols like Render/Akash allow people to rent idle GPU power for AI training.\n"
                       "- **Data Orchestration**: The Graph/Bittensor allow for decentralized 'searching' and 'training' of models, preventing AI from being a 'monopoly' of Big Tech.\n"
                       "- **FHE & Privacy**: Advanced encryption (Fully Homomorphic Encryption) allows AI models to process encrypted data without ever seeing the raw sensitive info.\n"
                       "- **Bull Case**: One of the strongest narratives due to the 'AI Revolution' in the real world. High correlation with Nvidia stock price.\n"
                       "- **Primary Examples**: Bittensor (TAO), Render (RNDR), Fetch.ai (FET), SingularityNET.",
        
        'cat_stable_title': "Stablecoins - The Global Dollar Rail",
        'cat_stable_desc': "**Stablecoins** act as the medium of exchange and 'Safe Haven' within the crypto ecosystem.\n\n"
                           "**Deep Dive Analysis**:\n"
                           "- **Collateral Models**:\n"
                             "  - *Fiat-Backed*: 1:1 backed by real dollars in bank accounts (e.g., USDT, USDC). Subject to central counterparty risk/regulation.\n"
                             "  - *Over-Collateralized*: Backed by other crypto (e.g., DAI). 1 dollar is backed by $1.50 of ETH to absorb volatility.\n"
                             "  - *Algorithmic*: Depend on supply/demand math. Extremely high-risk (reminder: Terra/Luna collapse).\n"
                           "- **Role in Liquidity**: Stablecoins are the most liquid pairs. They provide the 'Fuel' for every other market sector.\n"
                           "- **Yield Basis**: Often the safest way to earn yield through Lending or LPing on stable pairs.\n"
                           "- **Primary Examples**: Tether (USDT), Circle (USDC), DAI, Ethena (USDe).",
        
        'cat_alt_title': "Altcoins - The Diversification Layer",
        'cat_alt_desc': "**Altcoins** (Alternative Coins) refer to any cryptocurrency other than Bitcoin. This broad category includes everything from major smart contract platforms to tiny experimental tokens.\n\n"
                        "**Deep Dive Analysis**:\n"
                        "- **Market Correlation**: Altcoins typically have a high positive correlation with Bitcoin but with higher volatility (higher Beta). During 'Altseason', Altcoins often outperform Bitcoin significantly.\n"
                        "- **The Innovation Lab**: Alts are where the most radical experiments happen—new consensus models, privacy features, and complex financial instruments.\n"
                        "- **Risk Strategy**: Investors use Altcoins to find 'Alpha' (excess return). While Bitcoin is 'Digital Gold' for capital preservation, Altcoins are 'Growth Tech' for capital appreciation.\n"
                        "- **Cycle Dynamics**: Funds typically flow from Bitcoin -> Ethereum -> Large Cap Alts -> Mid/Small Cap Alts. Tracking this rotation is the key to timing market exits.",
        
        # New How To (EN)
        'howto_step1': "1. **Scan**: Use 'Crypto Scanner' to find coins with High Scores (>70) and Low Risk.",
        'howto_step2': "2. **Analyze**: Check 'Deep Dive' to see if whales are buying (Volume Growth).",
        'howto_step3': "3. **Portfolio**: Use 'Auto-Wealth' to allocate size based on safety.",

        
        # UI Labels
        'ui_capital': "Capital Amount (USD)",
        'ui_risk': "Risk Tolerance",
        'ui_generate': "Generate Optimal Portfolio",
        'ui_results': "Results",
        
        
        # --- Navigation Keys ---
        'nav_scanner': "Crypto Scanner",
        'nav_single': "Single Coin Analysis",
        'nav_wealth': "Auto-Wealth",
        'nav_glossary': "Crypto Glossary",
        'nav_howto': "How to Use",
        
        'nav_howto': "How to Use",
        
        # --- Page Headers ---
        'scanner_header': "Bidnow Scan",
        'scanner_subtitle': "Institutional-Grade Crypto Screener powered by Bidnow Engine.",
        'deep_dive_title': "Bidnow Dept",
        'deep_dive_subtitle': "Deep-tier fundamental analysis and valuation modeling.",
        'wealth_title': "Bidnow Wealth",
        'wealth_subtitle': "Institutional-Grade Portfolio Construction using Modern Portfolio Theory (MPT).",
        'glossary_title': "Bidnow Glossary",
        'glossary_subtitle': "Technical definitions and quantitative methodology framework.",
        'howto_title': "Bidnow (Beta)",
        'howto_subtitle': "Technical framework and procedural guidelines for institutional analysis.",
        
        # --- Restored Keys --- 
        'tab_holdings': "📋 Holdings",
        'tab_alloc': "🍕 Allocation",
        'tab_logic': "⚖️ Weighting Logic",
        'risk_high_desc': "🚀 **Euphoria**: Chasing parabolic moves. High risk of bag-holding.",
        'menu_health': "Portfolio Health",
        'menu_ai': "AI Insight",
        'under_dev': "🚧 Feature Under Development 🚧",
        'dev_soon': "Check back soon!",
        'dev_dl': "Coming soon: Machine Learning Models.",
        'biz_summary': "📝 **Project Summary**",
        'lynch_type': "Narrative Type",
        'score_garp': "Cycle Score",
        'score_value': "Value Score",
        'score_div': "Yield Score",
        'score_multi': "Alpha Score",
        'market_sentiment_title': "### 🧭 Market Sentiment (CNN-Style Proxy)",
        'fear_greed_title': "Fear & Greed Index (Proxy)",
        'vix_caption': "Driven by VIX: {vix:.2f} (Lower VIX = Higher Greed)",
        'state_extreme_fear': "🥶 Extreme Fear",
        'state_fear': "😨 Fear",
        'state_neutral': "😐 Neutral",
        'state_greed': "😎 Greed",
        'state_extreme_greed': "🤑 Extreme Greed",
        'state_extreme_greed': "🤑 Extreme Greed",
        'faq_title': "📚 Definition & Methodology (FAQs)",
        'max_pe': "Max P/E Ratio",
        'max_peg': "Max PEG Ratio",
        'max_evebitda': "Max EV/EBITDA",
        'min_roe': "Min ROE %",
        'min_margin': "Min Op Margin %",
        'min_div': "Min Dividend Yield %",
        'min_rev_growth': "Min Revenue Growth %",
        'max_de': "Max Debt/Equity %", # Reserved
        'debug_logs': "🛠️ Debug Logs (Open if No Data)",
        'port_title': "Portfoliokub",
        'ai_analysis_header': "🧠 AI Analysis Result ({risk})",
        'gen_success': "✅ Generated Professional Portfolio: {n} Coins",
        
        # --- Additional English Keys ---
        'price_trend_title': "📉 5-Year Price Trend",
        'err_fetch': "Could not fetch data.",
        'perfect_match': "✅ Perfect Match",
        'backtest_summary': "Performance Summary",
        'final_val_label': "Final Portfolio Value",
        'bench_val_label': "S&P 500 Benchmark",
        'alpha_label': "Alpha (vs Market)",
        'winning': "Winning",
        'losing': "Losing",
        'gap_annual': "Performance Gap (Annual)",
        'my_port_legend': "My Portfolio",
        'bench_legend': "S&P 500 (SPY)",
        'cagr_label': "CAGR (Avg/Year)",
        'annualized_label': "Annualized",
        'na_short': "N/A (< 1 Year)",
        'na': "N/A",
        'backtest_failed': "Backtest Failed",
        'lang_label': "Language / ภาษา",
        'health_coming_soon': "Coming soon in Q1 2026. This module will analyze your upload portfolio for risk factors.",
        'ai_coming_soon': "Deep Learning module integration in progress.",
        'tab_settings': "🎛️ Settings & Tools",
        'tab_metrics': "📊 Financial Metrics",
        'tab_lynch': "🧠 Peter Lynch Categories",
        'port_alloc_title': "🌍 Portfolio Allocation",
        'port_alloc_caption': "Breakdown by Individual Holding & Group",
        'type_alloc_title': "Type Allocation",
        'equity_only': "Equity Only",
        'asset_class_label': "Asset Class",
        'sector_label_short': "Sector",
        'weight_label': "Weight",
        'ticker_label': "Symbol",
        'price_label': "Price",
        'score_label': "Score",
        'rev_cagr_label': "Rev CAGR",
        'ni_cagr_label': "NI CAGR",
        'yield_label': "Yield",
        'why_mcap_title': "**Why Market Cap Weighting?**",
        'why_mcap_desc': "- **Professional Standard**: S&P 500 and Nasdaq 100 use this.\n- **Stability**: Larger, more established companies get more money.\n- **Self-Correcting**: As companies grow, they become a larger part of your portfolio naturally.",
        'how_works_title': "**How it works here:**",
        'how_works_desc': "1. We select the Top 20 stocks that match your **Strategy Score**.\n2. We allocate money based on **Company Size (Market Cap)**.",
        'tab_glossary_metrics': "📊 Metrics & Logic",
        'tab_glossary_cats': "🪙 Asset Categories",
    },
    'TH': {
        'sidebar_title': "🏛️ แผงควบคุมสแกนเนอร์",
        'market_label': "หมวดหมู่คริปโต",
        'strategy_label': "กลยุทธ์การลงทุน",
        'mode_header': "3. โหมดการค้นหา",
        'strict_label': "เลือกเกณฑ์ที่ต้องการเน้น (Strict Logic)",
        'perf_label': "ตัวชี้วัดประสิทธิภาพ",
        'val_header': " ประเมินมูลค่า (On-Chain)",
        'prof_header': "⚡ โมเมนตัม (Technical)",
        'risk_header': "🛡️ ความเสี่ยง & ความผันผวน",
        'sector_label': "เลือกกลุ่มธุรกิจ (Sector)",
        'lynch_label': "เลือกวัฏจักรราคา (Cycle Phase)",
        'execute_btn': "🚀 เริ่มการสแกน (Bidnow Scan)",
        'main_title': "Bidnow (บิทนาว)",
        'scan_limit': "จำนวนเหรียญที่จะสแกน",
        'results_header': "🏆 ผลลัพธ์การค้นหา (วิเคราะห์เชิงลึก)",
        'stage1_msg': "📡 ขั้นแรก: กำลังดึงข้อมูลตลาด...",
        'stage2_msg': "✅ โหลดเสร็จสิ้น: กำลังวิเคราะห์ปัจจัยพื้นฐาน...",
        'no_data': "❌ ไม่พบเหรียญที่ตรงตามเงื่อนไข (ลองลดความเข้มงวดลง)",
        'deep_dive_title': "🔍 วิเคราะห์รายเหรียญ (Deep Dive)",
        'glossary_title': "📚 คลังความรู้คริปโต (Glossary)",
        'howto_title': "📘 คู่มือการใช้งาน",
        'search_ticker': "พิมพ์ชื่อเหรียญ (เช่น BTC-USD, ETH-USD)",
        'analyze_btn': "วิเคราะห์เหรียญนี้",
        'about_title': "ℹ️ เกี่ยวกับ Bidnow",
        'about_desc': "แพลตฟอร์มวิเคราะห์คริปโตระดับมืออาชีพ ใช้ทฤษฎีวัฏจักร (Cycle Theory) และข้อมูล On-Chain ในการหาจุดเข้าซื้อที่ได้เปรียบ",

        'scanner_config': "🛠️ ตั้งค่าสแกนเนอร์",
        'univ_scale': "1. ขอบเขตจักรวาล (Scale)",
        'strat_mandate': "2. เลือกกลยุทธ์ (Mandate)",
        'crit_thresh': "3. กำหนดเกณฑ์ (Thresholds)",
        'opt_filters': "ตัวกรองเสริม (Optional)",
        'analyze_top_n': "จำนวนเหรียญที่วิเคราะห์ลึก (Top N)",
        
        'port_config': "⚙️ ตั้งค่าพอร์ต", 
        'asset_univ': "1. เลือกสินทรัพย์",
        'strat_prof': "2. โปรไฟล์กลยุทธ์",
        'max_holdings': "จำนวนเหรียญสูงสุด",

        'gloss_mvrv': "**MVRV Z-Score (Market Value to Realized Value)**: ตัวบ่งชี้ระดับสถาบันที่ใช้ประเมินสภาวะตลาดสุดโต่ง (Market Extremes)\n\n"
                      "**ระเบียบวิธีวิเคราะห์**: คำนวณจากส่วนเบี่ยงเบนมาตรฐาน (Standard Deviation) ระหว่างมูลค่าตลาด (Market Cap) และต้นทุนจริง (Realized Cap) "
                      "หากค่าสูงกว่า 3.5 จะถูกพิจารณาว่าเป็นสภาวะ 'ราคาสูงเกินจริง' (Overvalued) ขณะที่ค่าต่ำกว่า 0.0 บ่งชี้สภาวะ 'ราคาต่ำกว่าพื้นฐาน' (Undervalued) "
                      "ซึ่งสะสมมาจากการกระจายตัวของต้นทุนผู้ถือครองในเครือข่าย",
        'gloss_rsi': "**RSI (Relative Strength Index)**: เครื่องมือคำนวณโมเมนตัมทางเทคนิครูปแบบ Oscillator สำหรับประเมินความเร็วและทิศทางของการเปลี่ยนแปลงราคา\n\n"
                     "**การประยุกต์ใช้**: ค่าดัชนีมีขอบเขตระหว่าง 0-100 โดยค่าที่สูงกว่า 70 บ่งชี้สภาวะการซื้อมากเกินไป (Overbought) ซึ่งมักนำไปสู่การปรับฐานราคา "
                     "และค่าที่ต่ำกว่า 30 บ่งชี้สภาวะการขายมากเกินไป (Oversold) ที่ราคาอาจมีการกลับตัวขึ้น",
        'gloss_sharpe': "**Sharpe Ratio**: มาตรวัดทางคณิตศาสตร์สำหรับประเมิน 'ผลตอบแทนที่ปรับด้วยความเสี่ยง' (Risk-Adjusted Return) "
                        "โดยคำนวณจากส่วนต่างของผลตอบแทนสินทรัพย์เทียบกับอัตราผลตอบแทนที่ปราศจากความเสี่ยง (Risk-free rate) หารด้วยค่าส่วนเบี่ยงเบนมาตรฐาน\n\n"
                        "**การตีความ**: ค่าที่สูงกว่า 1.0 บ่งชี้ถึงประสิทธิภาพในการบริหารพอร์ตโฟลิโอ โดยผลตอบแทนที่เพิ่มขึ้นไม่ได้มาจากความผันผวนที่เป็นอันตรายแต่มาจากความสามารถในการเลือกสินทรัพย์",
        'gloss_Bidnow_score': "**Bidnow Score (Institutional Standard)**: โมเดลเชิงปริมาณ (Quantitative Model) สำหรับประเมินความแข็งแกร่งปัจจัยพื้นฐาน โดยพิจารณาจาก 4 เสาหลัก:\n\n"
                               "1. **Financial Health (30%)**: วิเคราะห์กระแสรายได้ของโปรโตคอล (Revenue Generation) และความสมเหตุสมผลของราคาเมื่อเทียบกับรายได้ (P/S Ratio)\n"
                               "2. **Network Dynamics (30%)**: วิเคราะห์การเติบโตของธุรกรรมบนบล็อกเชน (On-chain Activity) โดยเปรียบเทียบปริมาณธุรกรรมปัจจุบันกับค่าเฉลี่ยย้อนหลัง\n"
                               "3. **Technological Infrastructure (20%)**: ประเมินระดับความเชื่อถือของระบบนิเวศและแผนการนำไปใช้จริง (Ecosystem Maturity)\n"
                               "4. **Supply-Side Dynamics (20%)**: ตรวจสอบโครงสร้างเศรษฐศาสตร์เหรียญ (Tokenomics) อัตราเงินเฟ้อ และกำหนดการปลดล็อคเหรียญเพื่อจำกัดความเสี่ยงจาก Dilution",
        'gloss_cycle': "**Cycle Risk Assessment**: มาตรวัดความเสี่ยงเชิงพยากรณ์สำหรับระบุตำแหน่งของสินทรัพย์ในวัฏจักรตลาด (Market Cycle Positions)\n\n"
                       "**ระดับการวิเคราะห์**: ระดับต่ำกว่า 20% บ่งชี้ถึงโซน 'โอกาสสะสมสูงสุด' (Low Risk) และกลุ่มระดับที่สูงกว่า 80% บ่งชี้ถึงโซน 'ความเสี่ยงสูงสุด' (Euphoria) "
                       "ซึ่งเป็นจุดที่มีความสัมพันธ์กับการเกิดการปรับฐานครั้งใหญ่ (Major Correction)",

        # --- Asset Categories (TH) ---
        'cat_l1_title': "Layer 1 (L1) - รากฐานมหาอำนาจบล็อกเชน",
        'cat_l1_desc': "**Layer 1 (L1)** คือเครือข่ายบล็อกเชนที่เป็นอิสระ มีระบบความปลอดภัยและกฎเกณฑ์ (Consensus) ของตัวเอง เปรียบเสมือน \"ประเทศ\" หรือ \"ระบบปฏิบัติการ\" ของโลกคริปโต\n\n"
                       "**การวิเคราะห์เชิงลึก (Deep Dive)**:\n"
                       "- **Scalability Trilemma**: ทุกเครือข่ายต้องสู้กับวิกฤต 3 ประการคือ ความปลอดภัย (Security), การกระจายศูนย์ (Decentralization) และความเร็ว (Scalability) การเลือกอย่างใดอย่างหนึ่งมักต้องเสียอีกอย่างไป เช่น Bitcoin เลือกความปลอดภัยสูงสุด แต่ช้าลง\n"
                       "- **โมเดลทางเศรษฐกิจ**: เหรียญ L1 คือ \"อสังหาริมทรัพย์ดิจิทัล\" มูลค่าของมันมาจากจำนวน dApps และผู้ใช้ที่มาสร้างบนเครือข่าย ยิ่งมีการใช้งานมาก ความต้องการเหรียญเพื่อจ่ายค่า Gas ก็ยิ่งสูงขึ้น\n"
                       "- **กลไกการปั๊มกำไร**: ในยุค Proof of Stake (PoS) การถือเหรียญ L1 ช่วยให้คุณได้รับ \"ปันผล\" ผ่านการ Staking ซึ่งเป็นรากฐานของรายได้แบบ Passive Income\n"
                       "- **ความเสี่ยง**: สงคราม L1 Wars มีความรุนแรง หากนักพัฒนาหนีไปใช้เชนที่เร็วกว่าหรือถูกกว่า มูลค่าเครือข่ายอาจลดลงอย่างรวดเร็ว\n"
                       "- **ตัวอย่าง**: Bitcoin (ทองคำดิจิทัล), Ethereum (คอมพิวเตอร์โลก), Solana (โครงสร้างพื้นฐานความเร็วสูง)",
        
        'cat_l2_title': "Layer 2 (L2) - กองทัพเสริมเพื่อการขยายตัว",
        'cat_l2_desc': "**Layer 2 (L2)** คือโปรโตคอลที่สร้างขึ้นบน Layer 1 (ส่วนใหญ่คือ Ethereum) เพื่อช่วยทำรายการให้เร็วขึ้นและถูกลง โดยยังคงพึ่งพาความปลอดภัยจากเครือข่ายหลัก\n\n"
                       "**การวิเคราะห์เชิงลึก (Deep Dive)**:\n"
                       "- **เทคโนโลยี Rollup**: คือการรวบรวมธุรกรรมจำนวนมากมาประมวลผลข้างนอกเครือข่ายหลัก แล้วส่ง \"หลักฐาน\" กลับไปบันทึกครั้งเดียว แบ่งเป็น:\n"
                         "  - *Optimistic Rollups*: เชื่อว่าทุกคนทำรายการถูกต้องก่อน ถ้ามีคนค้านค่อยตรวจสอบ (เช่น Arbitrum, Optimism)\n"
                         "  - *ZK-Rollups*: ใช้คณิตศาสตร์ขั้นสูงพิสูจน์ความถูกต้องทันที (Zero-Knowledge) ซึ่งมีความปลอดภัยและเป็นส่วนตัวสูงกว่า (เช่น ZK-Sync, Starknet)\n"
                       "- **ความปลอดภัยแบบ inheritance**: แม้ระบบ L2 จะล่ม แต่เงินของคุณยังคงปลอดภัยอยู่ใน L1 เสมอผ่านกลไกการถอนเงินฉุกเฉิน\n"
                       "- **โมเดลรายได้**: L2 ทำกำไรจากส่วนต่างของค่า Gas ที่เก็บจากผู้ใช้ เทียบกับค่าเช่าที่ต้องจ่ายให้ L1\n"
                       "- **ตัวอย่าง**: Arbitrum, Optimism, Base, Polygon",
        
        'cat_defi_title': "DeFi - ธนาคารโลกใหม่ไร้ตัวกลาง",
        'cat_defi_desc': "**Decentralized Finance (DeFi)** คือการยกเอาระบบการเงินทั้งหมดมาไว้บนบล็อกเชน โดยใช้โค้ดคอมพิวเตอร์ (Smart Contracts) ทำหน้าที่แทนธนาคาร\n\n"
                         "**การวิเคราะห์เชิงลึก (Deep Dive)**:\n"
                         "- **ระบบ AMM (Automated Market Maker)**: การเทรดแบบไม่ต้องง้อ Matching Order แต่เทรดกับ \"Liquidity Pool\" (สระสภาพคล่อง) ทำให้คุณเทรดได้ 24 ชั่วโมงโดยไม่มีวันหยุด\n"
                         "- **Money Legos (คอมโพสซาบิลิตี้)**: โปรโตคอลต่างๆ สามารถต่อยอดกันได้เหมือนเลโก้ เช่น คุณฝากเหรียญในโปรโตคอล A เพื่อกู้ แล้วเอาเงินกู้ไปฟาร์มต่อในโปรโตคอล B ทั้งหมดนี้จบได้ในธุรกรรมเดียว\n"
                         "- **Real Yield vs Inflation**: นักลงทุนสถาบันมุ่งเน้นไปที่โปรโตคอลที่มีรายได้จริงจากค่าธรรมเนียม (Real Yield) มากกว่าการแจกเหรียญฟรีที่ทำให้เกิดเงินเฟ้อ\n"
                         "- **ความเสี่ยง**: การถูก Hack ช่องโหว่ของโค้ด (Exploits) หรือความเสี่ยงจาก Oracle (ข้อมูลราคาผิดพลาดจนทำให้ถูกล้างพอร์ต)\n"
                         "- **ตัวอย่าง**: Uniswap (กระดานเทรด), Aave (ธนาคารกู้ยืม), MakerDAO (ระบบผลิต Stablecoin)",
        
        'cat_gamefi_title': "GameFi & Metaverse - เศรษฐกิจโลกเสมือน",
        'cat_gamefi_desc': "**GameFi** คือจุดตัดระหว่างเกมและการเงิน (Game + Finance) ที่ทำให้การเล่นเกมไม่ใช่แค่ความสนุก แต่คือการสร้างรายได้จริง\n\n"
                           "**การวิเคราะห์เชิงลึก (Deep Dive)**:\n"
                           "- **กรรมสิทธิ์ดิจิทัลที่แท้จริง**: ผ่านเทคโนโลยี NFT ไอเทมในเกมทุกชิ้น หรือที่ดินในโลกเสมือน คือทรัพย์สินที่คุณเป็นเจ้าของจริง และสามารถนำไปขายหรือค้ำประกันเงินกู้ได้\n"
                           "- **GDP ของโลกเสมือน**: ใน Metaverse มีการสร้างระบบเศรษฐกิจที่มีมูลค่าจริง มีการจ้างงาน การโฆษณา และการจัดคอนเสิร์ตที่เก็บค่าเข้าชมเป็นคริปโต\n"
                           "- **จาก P2E สู่ Play-and-Earn**: ยุคแรกเน้นปั่นเงินจนล่มสลาย แต่ยุคใหม่มุ่งเน้นความสนุกนำหน้า แล้วมีรางวัลเป็นผลพลอยได้ เพื่อความยั่งยืนของระบบ\n"
                           "- **พฤติกรรมราคา**: เป็นสินทรัพย์ที่มีค่า Beta สูงมาก (ผันผวนแรงกว่าตลาด) มักจะวิ่งแรงที่สุดในช่วงที่ตลาดเข้าสู่ภาวะโลภสูงสุด (Euphoria)\n"
                           "- **ตัวอย่าง**: Axie Infinity, The Sandbox, Illuvium, Gala Games",
        
        'cat_meme_title': "Memecoins - สินทรัพย์แห่งกระแสศรัทธา",
        'cat_meme_desc': "**Memecoins** คือเหรียญที่ขับเคลื่อนด้วยพลังของชุมชน วัฒนธรรมอินเทอร์เน็ต และกระแสสังคม มากกว่าพื้นฐานทางธุรกิจหรือเทคโนโลยี\n\n"
                         "**การวิเคราะห์เชิงลึก (Deep Dive)**:\n"
                         "- **Social Consensus**: พลังแห่งความเชื่อร่วมกัน หากคนนับล้านเชื่อว่ารูปหมามีมูลค่า มันก็จะมีมูลค่าตามกฎ Demand & Supply เป็นตัวแทนของ Attention Economy (เศรษฐกิจฐานความสนใจ)\n"
                         "- **Fair Launch**: หลายเหรียญเปิดตัวอย่างยุติธรรม ไม่มีเจ้ามือหรือ VC ถือเหรียญราคาถูก ทำให้ชุมชนมีความร่วมใจสูงและมีความเป็นลัทธิ (Cult) ในการถือครอง\n"
                         "- **Slot Machine Reflexivity**: มีแรงดึงดูดใจจากการสร้างผลตอบแทน 1,000 เท่า ทำให้เกิดวงจรสะท้อนกลับ ราคาขึ้น -> คนรุม -> ราคาขึ้นต่อ\n"
                         "- **ความอันตราย**: มีความเสี่ยงสูงที่จะกลายเป็นศูนย์ (Total Loss) มีสภาพคล่องต่ำ ถ้าราคาตกร่วงแรงอาจจะไม่มีคนรับซื้อ\n"
                         "- **ตัวอย่าง**: Dogecoin, Shiba Inu, Pepe, Dogwifhat",
        
        'cat_ai_title': "AI Protocols - พลังปัญญาประดิษฐ์ไร้ศูนย์กลาง",
        'cat_ai_desc': "**AI Protocols** คือโครงการที่ผสานบล็อกเชนเข้ากับ AI เพื่อให้เข้าถึงพลังประมวลผล (Compute) และข้อมูล (Data) ได้อย่างเสรี\n\n"
                       "**การวิเคราะห์เชิงลึก (Deep Dive)**:\n"
                       "- **GPU กระจายศูนย์**: แทนที่จะหวังพึ่งแค่ Nvidia หรือ Azure โปรโตคอลอย่าง Render ช่วยให้คนทั่วไปสามารถเช่าพลังการประมวลผลที่เหลืออยู่เพื่อใช้อบรม AI ได้\n"
                         "- **การต่อต้านการผูกขาด**: Bittensor ช่วยสร้างฐานข้อมูลขนาดใหญ่ที่ไม่มีใครคนใดคนหนึ่งเป็นเจ้าของ ป้องกันไม่ให้ AI ตกอยู่ใต้การควบคุมของบริษัท Big Tech เพียงอย่างเดียว\n"
                         "- **AI + Privacy**: ใช้เทคโนโลยีการเข้ารหัสขั้นสูง เพื่อให้ AI เรียนรู้จากข้อมูลได้โดยไม่ต้องเห็นข้อมูลส่วนตัวของผู้ใช้จริงๆ\n"
                         "- **The AI Supercycle**: เป็นกลุ่มที่ได้รับความสนใจสูงสุดตามกระแสโลกจริง มีความสัมพันธ์ (Correlation) สูงกับราคาหุ้นกลุ่มเซมิคอนดักเตอร์\n"
                         "- **ตัวอย่าง**: Bittensor (TAO), Render (RNDR), Fetch.ai (FET)",
        
        'cat_stable_title': "Stablecoins - เส้นเลือดใหญ่และความมั่งคั่ง",
        'cat_stable_desc': "**Stablecoins** คือเสาหลักของสภาพคล่อง เป็นสินทรัพย์ที่ใช้พักเงินและรักษามูลค่าในช่วงที่ตลาดผันผวน\n\n"
                           "**การวิเคราะห์เชิงลึก (Deep Dive)**:\n"
                           "- **โมเดลการค้ำประกัน**:\n"
                             "  - *Fiat-Backed*: ค้ำด้วยเงินจริง 1:1 ในธนาคาร (เช่น USDT, USDC) มั่นใจสุดแต่เสี่ยงเรื่องการตรวจสอบโดยภาครัฐ\n"
                             "  - *Over-Collateralized*: ค้ำด้วยคริปโตเหรียญอื่นเกินมูลค่า (เช่น DAI) เช่น ใช้ ETH มูลค่า 150 บาท เพื่อผลิตเหรียญ 100 บาท ปลอดภัยจากตัวกลาง\n"
                             "  - *Algorithmic*: ค้ำด้วยคณิตศาสตร์ล้วนๆ (เช่น UST ในอดีต) เสี่ยงสูงมากหากเกิดวิกฤตความเชื่อ\n"
                           "- **บทบาทในระบบ**: เป็นสกุลเงินหลักที่ใช้เทรดทุกเหรียญในโลก เป็นดัชนีวัดปริมาณเงินในระบบคริปโต (Stablecoin Inflow = ตลาดกระทิงกำลังมา)\n"
                           "- **การสร้างกระแสเงินสด**: เป็นวิธีที่ปลอดภัยที่สุดในการหา Yield ผ่านการเป็นผู้ให้สภาพคล่องในคู่เหรียญที่ราคาคงที่\n"
                           "- **ตัวอย่าง**: Tether (USDT), Circle (USDC), DAI, USDe",
        
        'cat_alt_title': "อัลคอยน์ (Altcoins) - เลเยอร์แห่งการกระจายการลงทุน",
        'cat_alt_desc': "**Altcoins** (Alternative Coins) คือชื่อเรียกเหรียญคริปโตเคอร์เรนซีทุกชนิดที่ไม่ใช่ Bitcoin หมวดหมู่นี้ครอบคลุมตั้งแต่โปรเจกต์ระดับโลกไปจนถึงเหรียญทดลองคริปโตใหม่ๆ\n\n"
                        "**การวิเคราะห์เชิงลึก (Deep Dive)**:\n"
                        "- **ความสัมพันธ์กับตลาด**: ปกติแล้ว Altcoins จะวิ่งตาม Bitcoin แต่จะมีความผันผวนที่สูงกว่ามาก (High Beta) ในช่วง 'Altseason' เหรียญกลุ่มนี้มักจะสร้างผลตอบแทนได้ดีกว่า Bitcoin หลายเท่าตัว\n"
                        "- **ห้องทดลองนวัตกรรม**: Altcoins คือพื้นที่สำหรับนวัตกรรมที่ล้ำหน้าที่สุด เช่น ระบบการปกครองแบบ DAO, กลไกความเป็นส่วนตัวขั้นสูง และตราสารทางการเงินซับซ้อน\n"
                        "- **กลยุทธ์ความเสี่ยง**: นักลงทุนใช้ Altcoins เพื่อหา 'Alpha' (ผลตอบแทนส่วนเกิน) หาก Bitcoin คือทองคำเพื่อรักษาความมั่งคั่ง Altcoins ก็คือหุ้นเทคโนโลยีเติบโตสูงเพื่อเพิ่มความมั่งคั่ง\n"
                        "- **วัฏจักรการไหลของเงิน**: เงินมักจะไหลจาก Bitcoin -> Ethereum -> เหรียญ Alts ขนาดใหญ่ -> เหรียญขนาดกลาง/เล็ก การจับจังหวะการหมุนเวียนนี้คือหัวใจของการทำกำไร",

        'ui_capital': "เงินทุนเริ่มต้น (USD)",
        'ui_risk': "ความเสี่ยงที่รับได้",
        'ui_generate': "คำนวณพอร์ตแนะนำ",
        'ui_results': "ผลลัพธ์การจัดพอร์ต",
        
        # --- TH Restored Keys (Mapped to EN for now to prevent missing key error, can translate later) ---
        'tab_holdings': "📋 รายการสินทรัพย์",
        'tab_alloc': "🍕 สัดส่วนพอร์ต",
        'tab_logic': "⚖️ ตรรกะการคำนวณ",
        'gen_success': "✅ สร้างพอร์ตสำเร็จ: {n} เหรียญ",
        
        # --- Legacy / Inherited Keys (kept to prevent missing key errors) ---
        'lynch_tooltip': "",
        'lynch_desc': "Cycle Phases (Wyckoff/Market Cycle):\n- Accumulation: Smart Money buying quietly.\n- Markup: Public participation phase.\n- Distribution: Smart Money selling.\n- Markdown: Price decline.",
        'sector_tooltip': "",
        'sector_desc': "Narrative Categories (e.g. L1, DeFi, GameFi). Capital rotates between narratives.",
        'backtest_title': "🕑 Historical Backtest & Simulation",
        'backtest_desc': "See how this portfolio would have performed in the past vs S&P 500.",
        'backtest_config': "⚙️ Backtest Configuration",
        'invest_mode': "Investment Mode",
        'time_period': "Time Period",
        'invest_amount': "Investment Amount",
        'run_backtest_btn': "🚀 Run Backtest",
        'historical_chart_title': "### 🔬 Interactive Historical Charts",
        'select_stock_view': "Select Coin to View:",
        'nav_scanner': "สแกนคริปโต",
        'nav_single': "วิเคราะห์รายตัว",
        'nav_wealth': "จัดพอร์ต",
        'nav_portfolio': "จัดพอร์ต",
        'nav_health': "สุขภาพพอร์ต",
        'nav_ai': "AI วิเคราะห์",
        'nav_glossary': "คลังคำศัพท์",
        'nav_howto': "คู่มือการใช้งาน",
        'nav_help': "คู่มือการใช้งาน", 
        'scanner_header': "Bidnow Scan",
        'scanner_subtitle': "สแกนเนอร์คริปโตระดับสถาบัน ขับเคลื่อนด้วยระบบ Bidnow Engine",
        'deep_dive_title': "Bidnow Dept",
        'deep_dive_subtitle': "นิเคราะห์ปัจจัยพื้นฐานเชิงลึกและการสร้างแบบจำลองมูลค่า",
        'wealth_title': "Bidnow Wealth",
        'wealth_subtitle': "การสร้างพอร์ตโฟลิโอระดับสถาบันด้วยทฤษฎี Modern Portfolio Theory (MPT)",
        'glossary_title': "Bidnow Glossary",
        'glossary_subtitle': "คำนิยามทางเทคนิคและกรอบระเบียบวิธีวิเคราะห์เชิงปริมาณ",
        'howto_title': "Bidnow(เบต้า)",
        'howto_subtitle': "กรอบแนวทางทางเทคนิคและระเบียบขั้นตอนการวิเคราะห์ระดับสถาบัน",
        'footer_caption': "Professional Crypto Analytics Platform",
        'health_check_title': "🔍 On-Chain Health Check",
        'val_label': "Valuation",
        'qual_label': "Quality",
        'no_target': "No analyst target price available.",
        'err_recs': "Could not fetch recommendations.",
        'tab_glossary_metrics': "📊 ตัวชี้วัดและตรรกะ",
        'tab_glossary_cats': "🪙 ประเภทคริปโต",
    }
}

def get_text(key):
    lang = st.session_state.get('lang', 'EN')
    return TRANS[lang].get(key, key)

# --- MARKET & GURU DATA ---

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_market_indicators():
    indicators = {}
    try:
        # 1. Crypto Fear & Greed (Alternative.me API)
        import requests
        try:
            r = requests.get("https://api.alternative.me/fng/", timeout=2)
            if r.status_code == 200:
                data = r.json()
                fng_val = int(data['data'][0]['value'])
                fng_class = data['data'][0]['value_classification']
                indicators['FG_Score'] = fng_val
                indicators['FG_Class'] = fng_class
        except:
             # Fallback to VIX Proxy if API fails
            vix = yf.Ticker("^VIX")
            vix_val = vix.fast_info.last_price
            score = 100 - ((vix_val - 12) / (35 - 12) * 100)
            indicators['FG_Score'] = int(max(0, min(100, score)))
            indicators['FG_Class'] = "Proxy (VIX)"

        # 2. Bitcoin Dominance Proxy (BTC Market Cap / Total - Hard to get Total from YF)
        # We'll use BTC Price Trend as "Cycle Strength"
        btc = yf.Ticker("BTC-USD")
        hist = btc.history(period="1y")
        if not hist.empty:
            current = hist['Close'].iloc[-1]
            ma200 = hist['Close'].rolling(200).mean().iloc[-1]
            indicators['Trend_Diff'] = ((current - ma200) / ma200) * 100
            
    except Exception as e:
        print(f"Market Data Error: {e}")
        
    return indicators

def render_market_dashboard():
    data = fetch_market_indicators()
    if not data: return 

    st.markdown(get_text('market_sentiment_title'))
    
    # --- ROW 1: FEAR & GREED + CYCLE ---
    c1, c2 = st.columns([1, 1])
    
    with c1:
        score = data.get('FG_Score', 50)
        state = data.get('FG_Class', 'Neutral')
        
        # Color Logic
        if score < 25: color = "red"
        elif score > 75: color = "green"
        else: color = "orange"
        
        st.metric(get_text('fear_greed_title'), f"{score}/100", state)
        st.progress(score / 100)
        st.caption("Source: Alternative.me")

    with c2:
        # Cycle Strength (BTC vs 200DMA)
        trend = data.get('Trend_Diff', 0)
        st.metric("Bitcoin Bull Market Support", f"{trend:+.1f}%", "Above 200 DMA" if trend > 0 else "Below Support")
        st.caption("Distance from 200-Day Moving Average. > 0% is Bullish.")
        if trend > 0: st.success("Bitcon is in a Bull Trend 🐂")
        else: st.error("Bitcoin is in a Bear/Correction Trend 🐻")

    # --- ROW 2: FAQs ---
    with st.expander(get_text('faq_title')):
        st.markdown("""
        **What is the Fear & Greed Index?**  
        It is a way to gauge stock market movements and whether stocks are fairly priced. The logic is that **excessive fear drives prices down** (opportunity), and **too much greed drives them up** (correction risk).

        **How is it Calculated? (Official vs Proxy)**  
        - *Official (CNN)*: Compiles 7 indicators (Momentum, Strength, Breadth, Options, Junk Bonds, Volatility, Safe Haven).  
        - *Our Proxy*: We rely primarily on **Volatility (VIX)** and **Market Momentum** due to real-time data availability.

        **Scale:**  
        - **0-25**: Extreme Fear 🥶  
        - **25-45**: Fear 😨  
        - **45-55**: Neutral 😐  
        - **55-75**: Greed 😎  
        - **75-100**: Extreme Greed 🤑
        """)



# --- DEFILLAMA HELPER ---
@st.cache_data(ttl=3600*12, show_spinner=False)
def fetch_defillama_fees():
    """
    Fetches Protocol Fees & Revenue from DeFiLlama.
    Returns a dict mapping 'symbol' -> {'revenue_yearly': float, 'revenue_daily': float}
    """
    url = "https://api.llama.fi/overview/fees?excludeTotalDataChart=true&excludeTotalDataChartBreakdown=true"
    out = {}
    try:
        import requests
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            data = r.json()
            if 'protocols' in data:
                for p in data['protocols']:
                    # Mapping: We need to match Ticker to their Symbol
                    # DeFiLlama uses 'symbol' e.g. "BTC"
                    sym = p.get('symbol')
                    if not sym: continue
                    
                    # Extract Metrics
                    # 'total24h' is daily fees. 'total1y' is yearly fees.
                    # Note: For some protocols Fees = Revenue (like Uniswap LPs), for others (like Maker) it differs.
                    # We'll stick to 'total1y' as a proxy for "Economic Value" generated.
                    
                    rev_1y = p.get('total1y', 0)
                    rev_24h = p.get('total24h', 0)
                    
                    # Normalize simple symbol
                    out[sym.upper()] = {
                        'revenue_yearly': rev_1y if rev_1y else 0,
                        'revenue_daily': rev_24h if rev_24h else 0
                    }
    except Exception as e:
        print(f"DeFiLlama Error: {e}")
    
    return out

# ---------------------------------------------------------
# 1. Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Bidnow",
    page_icon="🪙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional Look
st.markdown("""
    <style>
    /* .stMetric removed for Dark Mode compatibility */
    .stDataFrame {
        font-family: 'IBM Plex Mono', 'Consolas', monospace;
        font-size: 0.95rem;
    }
    div[data-testid="stExpander"] div[role="button"] p {
        font-size: 1.1rem;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. Data Caching & Fetching
# ---------------------------------------------------------



# --- CRYPTO UNIVERSE DATA ---
@st.cache_data(ttl=86400)
def get_crypto_universe(category='Top 50'):
    """
    Returns a list of Yahoo Finance tickers for Cryptocurrencies.
    Examples: 'BTC-USD', 'ETH-USD'
    """
    
    # 1. Top 50 (Market Cap Proxy)
    top_50 = [
        'BTC-USD', 'ETH-USD', 'BNB-USD', 'SOL-USD', 'XRP-USD', 'DOGE-USD', 'ADA-USD', 'SHIB-USD', 
        'AVAX-USD', 'TRX-USD', 'DOT-USD', 'LINK-USD', 'MATIC-USD', 'UNI-USD', 'LTC-USD', 
        'BCH-USD', 'ATOM-USD', 'XLM-USD', 'ETC-USD', 'XMR-USD', 'FIL-USD', 'HBAR-USD', 
        'APT-USD', 'CRO-USD', 'LDO-USD', 'ARB-USD', 'NEAR-USD', 'VET-USD', 'MKR-USD', 
        'QNT-USD', 'AAVE-USD', 'GRT-USD', 'ALGO-USD', 'STX-USD', 'SAND-USD', 'EGLD-USD', 
        'THETA-USD', 'FTM-USD', 'EOS-USD', 'MANA-USD', 'FLOW-USD', 'AXS-USD', 'NEO-USD',
        'XTZ-USD', 'KCS-USD', 'CHZ-USD', 'GALA-USD', 'KLAY-USD', 'RUNE-USD', 'CRV-USD',
        # Top 100-200 Extension
        'HBAR-USD', 'VET-USD', 'ICP-USD', 'FIL-USD', 'EGLD-USD', 'MANA-USD', 'SAND-USD',
        'AXS-USD', 'THETA-USD', 'EOS-USD', 'AAVE-USD', 'FLOW-USD', 'QNT-USD', 'GRT-USD',
        'SNX-USD', 'NEO-USD', 'XEC-USD', 'MKR-USD', 'KLAY-USD', 'GNO-USD', 'CAKE-USD',
        'CFX-USD', 'ROSE-USD', 'WOO-USD', 'LUNC-USD', 'ZEC-USD', 'IOTA-USD', 'DASH-USD',
        'COMP-USD', 'FXS-USD', 'LRC-USD', 'ZIL-USD', 'DYDX-USD', 'CVX-USD', 'ENJ-USD',
        'BAT-USD', 'TWT-USD', 'MINA-USD', 'RVN-USD', 'XEM-USD', '1INCH-USD', 'HOT-USD',
        'GLM-USD', 'CELO-USD', 'KSM-USD', 'NEXO-USD', 'BAL-USD', 'JASMY-USD', 'AR-USD',
        'QTUM-USD', 'ANKR-USD', 'TFUEL-USD', 'ONT-USD', 'KAVA-USD', 'ILV-USD', 'GMT-USD',
        'YFI-USD', 'MASK-USD', 'JST-USD', 'GLMR-USD', 'WBTC-USD', 'BTT-USD', 'SXP-USD'
    ]

    # ... existing categories ...
    l1 = [
        'BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD', 'AVAX-USD', 'TRX-USD', 'DOT-USD', 
        'ATOM-USD', 'NEAR-USD', 'ALGO-USD', 'FTM-USD', 'SUI-USD', 'SEI-USD', 'TIA-USD',
        'APT-USD', 'INJ-USD', 'KAS-USD', 'TON-USD', 'MINA-USD', 'HBAR-USD'
    ]
    
    # Combined Large List (~250+ Tickers covering Bitkub Major, L1, DeFi, GameFi, Meme, AI)
    all_market = list(set(top_50 + l1 + [
        # Major & Stable (Bitkub Group 1-2)
        'BTC-USD', 'ETH-USD', 'BCH-USD', 'XRP-USD', 'LTC-USD', 'BNB-USD', 'USDT-USD', 'USDC-USD', 'DAI-USD',
        # L1/Platform (Group 3)
        'ADA-USD', 'SOL-USD', 'DOT-USD', 'AVAX-USD', 'NEAR-USD', 'MATIC-USD', 'OP-USD', 'ARB-USD', 'TRX-USD', 
        'FTM-USD', 'ATOM-USD', 'SUI-USD', 'SEI-USD', 'IOST-USD', 'ZIL-USD', 'HBAR-USD', 'CELO-USD',
        # Meme (Group 4)
        'DOGE-USD', 'SHIB-USD', 'BONK-USD', 'PEPE-USD', 'FLOKI-USD', 'MEME-USD', 
        # GameFi (Group 5)
        'SAND-USD', 'MANA-USD', 'GALA-USD', 'AXS-USD', 'ENJ-USD', 'ILV-USD', 'APE-USD', 'BLUR-USD', 'CHZ-USD',
        # DeFi (Group 6)
        'LINK-USD', 'UNI-USD', 'AAVE-USD', 'CRV-USD', 'MKR-USD', 'COMP-USD', 'SUSHI-USD', 'BAND-USD',
        # AI & New (Group 7)
        'TAO-USD', 'RNDR-USD', 'WLD-USD', 'IMX-USD', 'LDO-USD', 'INJ-USD', 'DYDX-USD', 'GRT-USD', 'LUNC-USD',
        # Extended Market (Top 100-300 Fillers)
        'STX-USD', 'FIL-USD', 'VET-USD', 'QNT-USD', 'THETA-USD', 'EOS-USD', 'FLOW-USD', 'EGLD-USD', 'XTZ-USD', 'KCS-USD',
        'RUNE-USD', 'FXS-USD', 'KAVA-USD', 'MINA-USD', 'GNO-USD', '1INCH-USD', 'WOO-USD', 'ROSE-USD', 'AGIX-USD', 'FET-USD',
        'OCEAN-USD', 'AKT-USD', 'STRK-USD', 'ORDI-USD', 'TIA-USD', 'KAS-USD', 'TON-USD', 'XLM-USD', 'XMR-USD', 'ETC-USD',
        'BGB-USD', 'LEO-USD', 'OKB-USD', 'CRO-USD', 'MNT-USD', 'BSV-USD', 'ALGO-USD', 'BEAM-USD', 'ASTR-USD', 'GLM-USD',
        'LRC-USD', 'BAT-USD', 'TWT-USD', 'CVX-USD', 'BAL-USD', 'YFI-USD', 'ZEC-USD', 'IOTA-USD', 'NEO-USD', 'DASH-USD',
        'QTUM-USD', 'XEM-USD', 'RVN-USD', 'HOT-USD', 'ZRX-USD', 'ANKR-USD', 'ICX-USD', 'WAVES-USD', 'OMG-USD', 'SC-USD',
        # Top 300 Expansion
        'WIF-USD', 'JUP-USD', 'PYTH-USD', 'ORCA-USD', 'RAY-USD', 'JTO-USD', 'ONDO-USD', 'PENDLE-USD', 'ENA-USD', 'ETHFI-USD',
        'ZK-USD', 'ZRO-USD', 'BLAST-USD', 'MODE-USD', 'SAFE-USD', 'OSMO-USD', 'JUNO-USD', 'KUJI-USD', 'AXL-USD', 'STRD-USD',
        'NTRN-USD', 'SAGA-USD', 'DYM-USD', 'ALT-USD', 'MANTA-USD', 'XAI-USD', 'PIXEL-USD', 'PORTAL-USD', 'AEVO-USD', 'VANRY-USD',
        'RON-USD', 'MAVIA-USD', 'PRIME-USD', 'GME-USD', 'MOG-USD', 'TURBO-USD', 'BRETT-USD', 'DEGEN-USD', 'TOSHI-USD', 'COQ-USD',
        'MYRO-USD', 'SLERF-USD', 'BOME-USD', 'MEW-USD', 'WEN-USD', 'POPCAT-USD', 'GIGA-USD', 'MICHI-USD', 'MOTHER-USD', 'DADDY-USD',
        'TRUMP-USD', 'BODEN-USD', 'TREMP-USD', 'KOL-USD', 'ZBU-USD', 'NOT-USD', 'IO-USD', 'ATH-USD', 'SPEC-USD', 'DRIFT-USD',
        'KMNO-USD', 'TNSR-USD', 'W-USD', 'PARCL-USD', 'ZEUS-USD', 'SHDW-USD', 'CLOUD-USD', 'MOBILE-USD', 'HONEY-USD', 'HNT-USD',
        'IOT-USD', 'DATA-USD', 'SUPER-USD', 'ERN-USD', 'HIGH-USD', 'TVK-USD', 'POLIS-USD', 'ATLAS-USD', 'STARL-USD', 'UFO-USD',
        'XYO-USD', 'LCX-USD', 'NMR-USD', 'TRB-USD', 'API3-USD', 'DIA-USD', 'UMA-USD', 'BADGER-USD', 'BOND-USD', 'FORTH-USD',
        # Maximizing List (Top 300-500 Candidates)
        'SNT-USD', 'CIVIC-USD', 'LOOM-USD', 'REQ-USD', 'POWR-USD', 'OXT-USD', 'ALICE-USD', 'DAR-USD', 'TLM-USD', 'ATA-USD',
        'BNT-USD', 'KNC-USD', 'REN-USD', 'STORJ-USD', 'BLZ-USD', 'COTI-USD', 'DENT-USD', 'DOCK-USD', 'DUSK-USD', 'ELF-USD',
        'FUN-USD', 'GAS-USD', 'IRIS-USD', 'KEY-USD', 'LTO-USD', 'MBL-USD', 'MDT-USD', 'MTL-USD', 'NKN-USD', 'NULS-USD',
        'PROS-USD', 'QUICK-USD', 'RARE-USD', 'REEF-USD', 'STPT-USD', 'STRAX-USD', 'STX-USD', 'SUN-USD', 'SUPER-USD', 'SYS-USD',
        'T-USD', 'TKO-USD', 'TOMO-USD', 'TRU-USD', 'UNFI-USD', 'VIDT-USD', 'VITE-USD', 'WAN-USD', 'WING-USD', 'WNXM-USD',
        'XNO-USD', 'XVG-USD', 'YGG-USD', 'YFII-USD', 'ZRX-USD', 'PROM-USD', 'PHA-USD', 'PERP-USD', 'ORN-USD', 'OGN-USD',
        'OCEAN-USD', 'NWC-USD', 'NUSE-USD', 'MOVR-USD', 'MLN-USD', 'MBOX-USD', 'LIT-USD', 'KSM-USD', 'KDA-USD', 'JASMY-USD',
        'HIVE-USD', 'HBAR-USD', 'GTC-USD', 'GNO-USD', 'GLM-USD', 'GHST-USD', 'FRONT-USD', 'FIDA-USD', 'FET-USD', 'FARM-USD',
        'DODO-USD', 'DGB-USD', 'DF-USD', 'CVC-USD', 'CTSI-USD', 'CTK-USD', 'C98-USD', 'BTM-USD', 'BTS-USD', 'BSV-USD',
        'BNX-USD', 'BICO-USD', 'BEL-USD', 'AUTO-USD', 'ATA-USD', 'ARPA-USD', 'ARDR-USD', 'ANT-USD', 'ALPACA-USD', 'ALICE-USD',
        'AKRO-USD', 'ADX-USD', 'ACH-USD', 'ACM-USD', 'ACA-USD', 'A8-USD', 'AERGO-USD', 'AGLD-USD', 'AION-USD', 'AIR-USD'
    ]))

    # Force unique 
    all_market = sorted(list(set(all_market)))

    # Categories
    l1 = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD', 'AVAX-USD', 'DOT-USD', 'NEAR-USD', 'MATIC-USD', 'ATOM-USD']
    defi = ['UNI-USD', 'AAVE-USD', 'MKR-USD', 'WIF-USD', 'LDO-USD', 'CRV-USD', 'LINK-USD']
    meme = ['DOGE-USD', 'SHIB-USD', 'PEPE-USD', 'FLOKI-USD', 'BONK-USD']
    ai_coins = ['RNDR-USD', 'TAO-USD', 'FET-USD', 'AGIX-USD', 'WLD-USD', 'GRT-USD']
    
    if category == 'Layer 1': return l1
    if category == 'DeFi': return defi
    if category == 'Meme': return meme
    if category == 'AI & Big Data': return ai_coins
    if category == 'All (Top 200)': return all_market
    
    # Default to Top 50 (Slice of all market)
    return list(all_market)[:50]


# --- CRYPTO METRIC HELPERS ---
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(close, fast=12, slow=26, signal=9):
    exp1 = close.ewm(span=fast, adjust=False).mean()
    exp2 = close.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    return macd, signal_line, hist

def calculate_atr(high, low, close, period=14):
    high_low = high - low
    high_close = (high - close.shift()).abs()
    low_close = (low - close.shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(period).mean()

def calculate_adx(high, low, close, period=14):
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    
    tr = calculate_atr(high, low, close, period=1) # TR for ADX calc
    atr = tr.rolling(period).mean()
    
    plus_di = 100 * (plus_dm.ewm(alpha=1/period).mean() / atr)
    minus_di = 100 * (minus_dm.abs().ewm(alpha=1/period).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(period).mean()
    return adx

def calculate_mvrv_z_proxy(series_price, series_vol, window=365):
    """
    Advanced Proxy: Uses VWAP (Volume Weighted Average Price) as 'Realized Price'.
    MVRV = Market Price / Realized Price (VWAP).
    Z-Score = (Market Cap - Realized Cap) / StdDev(Market Cap).
    Here we use Price level proxy.
    """
    # Rolling VWAP
    # VWAP = Sum(Price * Vol) / Sum(Vol)
    pv = series_price * series_vol
    
    # We use a rolling window to simulate "Active Cost Basis" (e.g. 1 Year)
    # Real Realized Cap is lifetime, but strictly that just trends up. 
    # A Rolling VWAP captures the "Recent Cost Basis" better for trading cycles.
    rolling_pv = pv.rolling(window=window).sum()
    rolling_vol = series_vol.rolling(window=window).sum()
    
    vwap = rolling_pv / rolling_vol
    
    # MVRV Ratio
    mvrv_ratio = series_price / vwap
    
    # Z-Score Calculation (Standardize the Ratio)
    # We want to know how far the CURRENT ratio is from the HISTORICAL average ratio
    mean_ratio = mvrv_ratio.rolling(window=window).mean()
    std_ratio = mvrv_ratio.rolling(window=window).std()
    
    z_score = (mvrv_ratio - mean_ratio) / std_ratio
    
    return z_score, mvrv_ratio

def detect_whale_activity(hist):
    """
    Analyzes Volume Spikes and Price Action to detect 'Smart Money'.
    Returns: Score (0-100), Description
    """
    if len(hist) < 30: return 0, "Insuff. Data"
    
    # 1. Volume Anomaly
    vol = hist['Volume']
    vol_ma_30 = vol.rolling(30).mean()
    vol_std_30 = vol.rolling(30).std()
    
    current_vol = vol.iloc[-1]
    last_ma = vol_ma_30.iloc[-1]
    last_std = vol_std_30.iloc[-1]
    
    if last_std == 0: return 0, "Stable"
    
    z_vol = (current_vol - last_ma) / last_std
    
    # 2. Price Action (Absorption Check)
    # If Vol is Huge but Price is Flat or Up = Accumulation
    # If Vol is Huge and Price is Down = Dumping
    close = hist['Close']
    open_ = hist['Open']
    price_change = (close.iloc[-1] - open_.iloc[-1]) / open_.iloc[-1]
    
    score = 50
    status = "Normal"
    
    if z_vol > 3.0: # Huge Spike (> 3 Sigma)
        if abs(price_change) < 0.01: # High Vol, Flat Price (Absorption/Handover)
            score = 95
            status = "🐋 WHALE ALERT: Absorption (High Vol, Flat Price)"
        elif price_change > 0:
            score = 85
            status = "🚀 WHALE ALERT: Aggressive Buying"
        else:
            score = 20 # Panic Dump
            status = "⚠️ WHALE ALERT: Panic Selling / Dumping"
    elif z_vol > 2.0:
        if price_change >= 0:
            score = 75
            status = "👀 Institutional Interest (Rising Vol)"
        else:
            score = 40
            status = "📉 High Selling Pressure"
            
    return score, status

# --- Stage 1: Fast Scan (Batch) ---
def scan_market_basic(tickers, progress_bar, status_text, debug_container=None):
    data_list = []
    status_text.text("Stage 1: Fetching Market Data (Batch Mode)...")
    
    if not tickers: return pd.DataFrame()
    
    import pandas as pd
    
    # Batch Download for Speed
    try:
        data = yf.download(tickers, period="2y", group_by='ticker', threads=True)
    except Exception as e:
        status_text.error(f"Download Failed: {e}")
        return pd.DataFrame()

    status_text.text("Stage 1: Calculating On-Chain Metrics...")
    
    total = len(tickers)
    
    # Handle Single Ticker vs Multi Ticker Structure
    if len(tickers) == 1:
        ticker = tickers[0]
        valid_tickers = [ticker]
    else:
        valid_tickers = tickers
        
    for i, ticker in enumerate(valid_tickers):
        if i % 5 == 0: progress_bar.progress((i + 1) / total)
        
        try:
            # Extract Series
            if len(tickers) == 1:
                hist = data
            else:
                hist = data[ticker]
            
            if hist is None or hist.empty or 'Close' not in hist.columns: continue
            
            # Drop NaN
            hist = hist.dropna(subset=['Close'])
            if len(hist) < 60: continue # Need history for VWAP
            
            # --- CALCULATE METRICS ---
            closes = hist['Close']
            vols = hist['Volume']
            
            # 1. Valuation: MVRV Z-Score Proxy (ADVANCED VWAP)
            z_score_series, _ = calculate_mvrv_z_proxy(closes, vols)
            current_z = z_score_series.iloc[-1] if not pd.isna(z_score_series.iloc[-1]) else 0
            
            # 2. Momentum: RSI
            rsi_series = calculate_rsi(closes)
            current_rsi = rsi_series.iloc[-1] if not pd.isna(rsi_series.iloc[-1]) else 50
            
            # 3. Whale Radar
            whale_score, whale_status = detect_whale_activity(hist)
            
            # 4. Volatility (30D)
            returns = closes.pct_change()
            vol_30d = returns.rolling(30).std().iloc[-1] * (365 ** 0.5) * 100
            if pd.isna(vol_30d): vol_30d = 0
            
            # 5. Cycle State
            cycle_state = "😐 Neutral"
            if current_z < -0.5: cycle_state = "🟢 Accumulation (Undervalued)" # VWAP is tighter, so lower Z threshold
            elif current_z > 3.0: cycle_state = "🔴 Euphoria (Overvalued)"
            elif current_z > 1.5: cycle_state = "🟠 Greed"
            
            narrative = classify_narrative(ticker)
            
            # 6. Price Change
            price = closes.iloc[-1]
            chg_7d = (price - closes.iloc[-8]) / closes.iloc[-8] * 100 if len(closes) > 7 else 0
            chg_30d = (price - closes.iloc[-31]) / closes.iloc[-31] * 100 if len(closes) > 31 else 0
            
            # --- PRO SCORE CALCULATION (Centralized Expert Engine) ---
            try:
                scores = calculate_Bidnow_score(ticker, hist, info=None)
                total_pro_score = scores.get('total', 0)
                analysis_str = get_grade(total_pro_score)
            except Exception as e:
                total_pro_score = 0
                analysis_str = "Error"
                scores = {} 
                
            # --- Bidnow LINE & MARGIN OF SAFETY ---
            try:
                c_line_series = calculate_Bidnow_line(hist)
                if not c_line_series.empty:
                    fair_value = c_line_series.iloc[-1]
                    mos = (fair_value - price) / price * 100 
                else:
                    fair_value = price
                    mos = 0
            except:
                fair_value = price
                mos = 0
            
            data_list.append({
                'Symbol': ticker,
                'Narrative': narrative,
                'Price': price,
                'Bidnow_Score': total_pro_score, 
                'Pro_Rating': analysis_str,
                'Fair_Value': fair_value,
                'Margin_Safety': mos,
                'MVRV_Z': current_z,
                'RSI': current_rsi,
                'Vol_30D': vol_30d,
                'Cycle_State': cycle_state,
                'Whale_Score': whale_score,
                'Whale_Status': whale_status,
                '7D': chg_7d,
                '30D': chg_30d,
                'YF_Obj': None 
            })
            
        except Exception as e:
            if debug_container: debug_container.write(f"Error {ticker}: {e}")
            continue
            
        except Exception as e:
            if debug_container: debug_container.write(f"Error {ticker}: {e}")
            continue
            
    if not data_list:
        return pd.DataFrame()
        
    return pd.DataFrame(data_list)



def analyze_history_deep(df_candidates, progress_bar, status_text):
    """
    Takes the surviving candidates and pulls history for deeper insight strings
    """
    total = len(df_candidates)
    enhanced_data = []
    
    for i, (idx, row) in enumerate(df_candidates.iterrows()):
        progress = (i + 1) / total
        progress_bar.progress(progress)
        ticker = row['Symbol']
        stock = row['YF_Obj']
        status_text.caption(f"Stage 2: Deep Analysis of **{ticker}** ({i+1}/{total})")
        
        # Metrics
        consistency_str = "N/A"
        insight_str = ""
        cagr_rev = None
        cagr_ni = None
        div_streak_str = "None"

        try:
            # Price Performance (NEW)
            hist = stock.history(period="5y")
            perf = {}
            if not hist.empty:
                # FIX: TZ awareness issues. Convert to naive.
                try:
                    hist.index = hist.index.tz_localize(None)
                except: pass
                
                curr_price = hist['Close'].iloc[-1]
                
                # Helper to get return
                def get_ret(days_ago):
                    try: 
                        # Use searchsorted to find closest date index
                        target_idx = hist.index.searchsorted(pd.Timestamp.now() - pd.Timedelta(days=days_ago))
                        if target_idx < len(hist):
                            old_price = hist['Close'].iloc[target_idx]
                            val = (curr_price - old_price) / old_price
                            return val * 100
                    except: pass
                    return None

                perf['1M'] = get_ret(30)
                perf['3M'] = get_ret(90)
                perf['6M'] = get_ret(180)
                perf['1Y'] = get_ret(365)
                perf['3Y'] = get_ret(365*3)
                perf['5Y'] = get_ret(365*5)
                
                # YTD
                current_year = pd.Timestamp.now().year
                ytd_start = hist[hist.index.year < current_year]
                if not ytd_start.empty:
                    ytd_price = ytd_start['Close'].iloc[-1]
                    perf['YTD'] = ((curr_price - ytd_price) / ytd_price) * 100
                else:
                    perf['YTD'] = None

        except Exception:
            div_streak_str = "Error"
            perf = {}
            pass
        
        # Build Data Dict
        data_item = {
            'Symbol': ticker,
            'Insight': insight_str if insight_str else "Stable"
        }
        # Merge perf metrics
        data_item.update(perf)
        enhanced_data.append(data_item)
        
    return pd.DataFrame(enhanced_data)

# ---------------------------------------------------------
# 3. Classifications & Scoring
# ---------------------------------------------------------
# ---------------------------------------------------------
# 3. Classifications & Scoring (Crypto Native)
# ---------------------------------------------------------
def classify_narrative(ticker):
    """
    Classifies coins into Crypto Narratives / Sectors.
    """
    t = ticker.upper()
    
    # 1. Store of Value
    if 'BTC' in t or 'PAXG' in t or 'XAUT' in t: return "👑 Store of Value"
    
    # 2. Smart Contracts (L1)
    l1s = ['ETH', 'SOL', 'ADA', 'BNB', 'AVAX', 'TRX', 'DOT', 'ATOM', 'NEAR', 'ALGO', 'SUI', 'SEI', 'APT', 'FTM']
    if any(x in t for x in l1s): return "🏗️ Smart Contract (L1)"
    
    # 3. DeFi
    defi = ['UNI', 'AAVE', 'MKR', 'LDO', 'CRV', 'SNX', 'COMP', 'RPL', 'GMX', 'DYDX', 'JUP']
    if any(x in t for x in defi): return "🏦 DeFi & Yield"
    
    # 4. Scaling (L2)
    l2s = ['MATIC', 'ARB', 'OP', 'IMX', 'MNT', 'STRK']
    if any(x in t for x in l2s): return "⚡ Layer 2 (Scaling)"
    
    # 5. Meme
    memes = ['DOGE', 'SHIB', 'PEPE', 'WIF', 'BONK', 'FLOKI', 'MEME']
    if any(x in t for x in memes): return "🤡 Meme / High Beta"
    
    # 6. AI & DePIN
    ai = ['RNDR', 'FET', 'TAO', 'AKT', 'OCEAN', 'AGIX', 'WLD']
    if any(x in t for x in ai): return "🤖 AI & DePIN"
    
    return "🪙 Altcoin"



# ---------------------------------------------------------
# PAGES
# ---------------------------------------------------------

# ---------------------------------------------------------
# STRATEGY PROFILES (Institutional Mandates)
# ---------------------------------------------------------
STRATEGY_PROFILES = {
    'Custom': {},
    '💎 Deep Value Gems': {
        'desc': 'Undervalued projects with strong revenue. (Buffett Style)',
        'roi': '+145%',
        'settings': {'mvrv_max': 0.5, 'score_min': 70, 'ps_max': 20}
    },
    '🚀 Network Growth': {
        'desc': 'High user growth and transaction volume. (Fisher Style)',
        'roi': '+210%',
        'settings': {'vol_growth_min': 20, 'score_min': 60}
    },
    '🐳 Whale Accumulation': {
        'desc': 'Smart money is buying while price is flat.',
        'roi': '+89%',
        'settings': {'vol_min': 5, 'vol_max': 40, 'rsi_max': 50} 
    },
    '🛡️ Risk-Adjusted Alpha': {
        'desc': 'Steady returns with low volatility.',
        'roi': '+65%',
        'settings': {'vol_max': 50, 'score_min': 80, 'dd_max': -20}
    },
    '💣 Contrarian Reversal': {
        'desc': 'Oversold coins ready for a bounce.',
        'roi': '+320%',
        'settings': {'rsi_max': 30, 'mvrv_max': 0, 'score_min': 50}
    }
}

def page_scanner():
    st.title(get_text('scanner_header'))
    st.caption(get_text('scanner_subtitle'))

    # --- 1. CONFIGURATION (Main Page) ---
    with st.expander("🛠️ **Scanner Configuration**", expanded=True):
        col_uni, col_strat = st.columns(2)
        
        with col_uni:
            st.subheader("1. Crypto Universe")
            
            # Helper to get count
            total_coins = len(get_crypto_universe('All (Top 200)'))
            
            market_choice = st.selectbox(f"Universe (Total: {total_coins} Coins)", ['All (Top 200)', 'Layer 1', 'DeFi', 'Meme', 'AI & Big Data'])
            scan_limit = st.slider("Max Coins to Scan", 10, total_coins, min(200, total_coins))
            
        with col_strat:
            st.subheader("2. Strategy Mandate")
            strat_choice = st.selectbox("Select Profile", list(STRATEGY_PROFILES.keys()))
            if strat_choice != 'Custom':
                roi = STRATEGY_PROFILES[strat_choice]['roi']
                st.caption(f"**Hist. ROI:** {roi} | {STRATEGY_PROFILES[strat_choice]['desc']}")
    
    # Pre-fill settings
    prof = STRATEGY_PROFILES[strat_choice].get('settings', {})
    
    # --- 2. CRITERIA THRESHOLDS ---
    st.subheader("📊 Screening Criteria")
    
    # A. Valuation & On-Chain
    with st.expander("A. Valuation & On-Chain (The 'Price')", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            filt_mvrv = st.slider("MVRV Z-Score (Max)", -5.0, 10.0, float(prof.get('mvrv_max', 3.5)), help="< 0 is Undervalued. > 3 is Overvalued.")
        with c2:
            filt_ps = st.slider("P/S Ratio (Max)", 0, 100, prof.get('ps_max', 100), help="Price to Sales. Lower is better value.")
        with c3:
            filt_nvt = st.slider("NVT Ratio (Max)", 0, 200, 150, help="Network Value to Transactions. Like P/E for Crypto.")

    # B. Financials & Quality
    with st.expander("B. Financials & Quality (The 'Good')"):
        c1, c2 = st.columns(2)
        with c1:
            filt_score = st.slider("Bidnow Score (Min)", 0, 100, prof.get('score_min', 40), help="0-100 Quality Score based on 4 pillars.")
        with c2:
            filt_vol_growth = st.slider("Vol Growth 30D (%) (Min)", -100, 500, prof.get('vol_growth_min', -100), help="Is usage growing?")

    # C. Technical & Pulse
    with st.expander("C. Technical & Pulse (The 'Timing')"):
        c1, c2, c3 = st.columns(3)
        with c1:
            filt_rsi = st.slider("RSI (Max)", 0, 100, prof.get('rsi_max', 100), help="< 30 Oversold, > 70 Overbought.")
        with c2:
            filt_vol = st.slider("Volatility 30D (Max)", 0, 200, prof.get('vol_max', 200), help="Lower = Safer.")
        with c3:
            # Placeholder for Social
            st.caption("Social Dominance: Not Available (API Limit)")

    # --- EXECUTE ---
    if st.button(f"🚀 Execute Scan ({market_choice})", type="primary"):
        tickers = get_crypto_universe(market_choice)
        tickers = tickers[:scan_limit]
        
        # UI Container
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Run Scan
        df_results = scan_market_basic(tickers, progress_bar, status_text)
        
        if df_results.empty:
            st.error("No data found. Check your internet or API limits.")
            return

        # --- CHECKLIST MATCHING LOGIC (Modified) ---
        # Instead of strict filtering, we calculate a "Match Score"
        
        def calculate_match(row):
            score = 0
            checks = []
            
            # 1. Bidnow Score
            if row['Bidnow_Score'] >= filt_score: 
                score += 1
                checks.append("✅ Score")
            
            # 2. MVRV
            if 'MVRV_Z' in row and row['MVRV_Z'] <= filt_mvrv:
                score += 1
                checks.append("✅ MVRV")
                
            # 3. RSI
            if 'RSI' in row and row['RSI'] <= filt_rsi:
                score += 1
                checks.append("✅ RSI")
                
            # 4. Volatility
            if 'Vol_30D' in row and row['Vol_30D'] <= filt_vol:
                score += 1
                checks.append("✅ Vol")
                
            return score, ", ".join(checks)

        # Apply Calculation
        df_results[['Scan_Score', 'Criteria_Met']] = df_results.apply(
            lambda x: pd.Series(calculate_match(x)), axis=1
        )
        
        # Apply Bidnow Ranking (Calculates Rank_Score but we will override Sort)
        df = calculate_Bidnow_ranking(df_results)
        
        # Sort by Scan Score DESC, then Bidnow Score DESC (Enforce Scan Priority)
        if not df.empty:
            df = df.sort_values(by=['Scan_Score', 'Bidnow_Score'], ascending=[False, False])

        st.markdown(f"### Results ({len(df)} Matches)")
        st.info("Ranking by Scan Score (Criteria Met).")


        
        # Color Styling for Cycle State
        # Color Styling for Cycle State & Rating
        def color_cycle(val):
            # Pro Rating Colors
            if isinstance(val, str):
                if "A" in val: return "color: #00FF00; font-weight: bold" # Green
                if "B" in val: return "color: lightgreen"
                if "D" in val or "F" in val: return "color: #ff4b4b"
                # Cycle Colors
                if "Accumulation" in val: return "color: lightgreen; font-weight: bold"
                if "Euphoria" in val: return "color: #ff4b4b; font-weight: bold"
                if "Greed" in val: return "color: orange"
            return ""
            
        def color_scan_score(val):
             return 'color: #00ccff; font-weight: bold' # Cyan

        # Columns to display
        # Added Bidnow_Score, Fair_Value, Margin_Safety
        display_cols = ['Symbol', 'Narrative', 'Scan_Score', 'Bidnow_Score', 'Pro_Rating', 'Price', 'Fair_Value', 'Margin_Safety', 'Cycle_State', '7D', '30D']
        
        st_df = df[display_cols].style.applymap(color_cycle, subset=['Cycle_State', 'Pro_Rating']) \
            .applymap(color_scan_score, subset=['Scan_Score']) \
            .format({
                'Price': '${:,.2f}',
                'Fair_Value': '${:,.2f}',
                'Margin_Safety': '{:.1f}%',
                '7D': '{:+.1f}%',
                '30D': '{:+.1f}%',
                'Bidnow_Score': '{:.0f}',
                'Scan_Score': '{:.0f}/4'   
            })
            
        st.dataframe(
            st_df,
            column_config={
                "Bidnow_Score": st.column_config.ProgressColumn("Bidnow Score", min_value=0, max_value=100, format="%d"),
                "Margin_Safety": st.column_config.NumberColumn("Margin of Safety", help="+ve: Undervalued, -ve: Overvalued"),
                "Fair_Value": st.column_config.NumberColumn("Wait-Wait Price", help="Intrinsic Value (Bidnow Line)"),
                "MVRV_Z": st.column_config.NumberColumn("On-Chain Z", help="< 0 is Buy")
            },
            hide_index=True,
            use_container_width=True
        ) 

        # --- Manual Deep Dive Section ---
        st.markdown("---")
        st.header("🔬 Interactive Historical Charts")
        st.info("Select a coin to visualize historical trends.")
        
        if 'Symbol' in df.columns:
            selected_ticker = st.selectbox("Select Ticker:", df['Symbol'].tolist(), index=0)
            
            # OPTION: Auto-display charts on selection (Better flow for user)
            # or use button. If button, we need to wrap it or it's fine now because parent blocks won't unrender
            if selected_ticker:
                with st.spinner(f"Pulling full history for {selected_ticker}..."):
                    # Use cached object if possible, or new fetch
                    # We stored YF_Obj in df, we can retrieve
                    try: # optimization
                        stock_obj = df.loc[df['Symbol'] == selected_ticker, 'YF_Obj'].values[0]
                    except:
                        stock_obj = None
                    
                    if stock_obj is None:
                        stock_obj = yf.Ticker(selected_ticker)
                    
                    
                    # CRYPTO: Show Price History instead of Financials
                    hist_data = stock_obj.history(period="1y")
                    if not hist_data.empty:
                        st.subheader(f"📈 {selected_ticker} Price Action (1Y)")
                        st.line_chart(hist_data['Close'])
                        
                        # Show Volume if available
                        if 'Volume' in hist_data.columns:
                            st.caption("Volume Trend")
                            st.bar_chart(hist_data['Volume'])
                    else:
                        st.warning("No price history available for this coin.")

        # Cache Clearing for Debugging
        # Cache Clearing for Debugging
        if st.checkbox("Show Advanced Options", key='adv_opt'):
            if st.button("🗑️ Clear Data Cache"):
                st.cache_data.clear()
                st.success("Cache Cleared! Rerun the scan.")
    
    elif st.session_state.get('scan_results') is None:
         # Only show this if no results AND no scan happening
         # But wait, if we are just idling, we don't want error.
         pass
         # st.info("Define parameters and start the Two-Stage Screening.")


def calculate_power_law_btc(days_since_genesis):
    """
    Giovanni Santostasi's Power Law for Bitcoin:
    Price = 10^-17 * (days)^5.8 roughly.
    We'll use a simplified fit for demo purposes or exact params if known.
    Model: Price = 10 ** ( -17.3 + 5.8 * log10(days) )
    Genesis: 2009-01-03
    """
    import math
    try:
        if days_since_genesis <= 0: return 0
        log_days = math.log10(days_since_genesis)
        # Parameters approximated from public charts
        log_price = -17.3 + 5.8 * log_days
        return 10 ** log_price
    except:
        return 0

def calculate_cycle_risk(current_price, ath):
    """
    Simple Risk Gauge: Dist form ATH.
    If Price ~= ATH, Risk is High (Local Top).
    If Price << ATH, Risk is Lower (Drawdown).
    """
    if not ath or ath == 0: return 0.5
    drawdown = (current_price - ath) / ath
    # Drawdown is negative e.g. -0.8
    # Risk Metric (0 to 1): 1 = At ATH (High Risk), 0 = -85% Down (Low Risk)
    
    # Map -0.85 (Low Risk) to 0.1
    # Map 0.0 (High Risk) to 0.9
    risk = 1.0 + drawdown # e.g. 1 + (-0.2) = 0.8
    return max(0.1, min(0.95, risk))

def calculate_stoch_rsi(series, period=14, smoothK=3, smoothD=3):
    """
    StochRSI = (RSI - MinRSI) / (MaxRSI - MinRSI)
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    stoch_rsi = (rsi - rsi.rolling(period).min()) / (rsi.rolling(period).max() - rsi.rolling(period).min())
    k = stoch_rsi.rolling(smoothK).mean() * 100
    d = k.rolling(smoothD).mean()
    return k, d

def calculate_cci(high, low, close, period=20):
    tp = (high + low + close) / 3
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: pd.Series(x).mad())
    cci = (tp - sma) / (0.015 * mad)
    return cci

# ---------------------------------------------------------
# PRO INTELLIGENCE SCORING (Startup Grade)
# ---------------------------------------------------------
# ---------------------------------------------------------
# PRO INTELLIGENCE SCORING (Bidnow Engine)
# ---------------------------------------------------------
def calculate_Bidnow_score(ticker, hist, info=None):
    """
    Bidnow SCORE A.I. (4 Pillars)
    1. Financial Health (30%) - Revenue & Valuation
    2. Network Activity (30%) - Usage & Volume
    3. Tech & Dev (20%) - Innovation (Simulated)
    4. Tokenomics (20%) - Supply & Inflation
    """
    
    score_cards = {
        'total': 0, 
        'financial': 0, 'network': 0, 'tech': 0, 'tokenomics': 0,
        'details': {'financial': [], 'network': [], 'tech': [], 'tokenomics': []},
        'analysis': []
    }
    
    # --- PREPARE DATA ---
    try:
        current_price = hist['Close'].iloc[-1]
        
        # Safe Info Access
        mcap = 0
        circ_supply = 0
        max_supply = 0
        vol_24h = hist['Volume'].iloc[-1]
        
        if info:
            mcap = info.get('marketCap', 0)
            circ_supply = info.get('circulatingSupply', 0)
            max_supply = info.get('maxSupply', 0)
        
        # Fallback Mcap if missing from info (Approximation)
        if mcap == 0 and circ_supply > 0:
            mcap = current_price * circ_supply
        
        # Clean Ticker for DeFiLlama (Remove -USD)
        clean_symbol = ticker.replace("-USD", "").upper()
            
        # ==============================================================================
        # 1. FINANCIAL HEALTH (30%)
        # Metrics: Revenue (DeFiLlama), P/S Ratio
        # ==============================================================================
        fin_score = 0
        fin_count = 0
        
        # A. Revenue Check
        fees_data = fetch_defillama_fees()
        coin_fees = fees_data.get(clean_symbol, {})
        rev_1y = coin_fees.get('revenue_yearly', 0)
        
        ps_ratio = 999
        if rev_1y > 0 and mcap > 0:
            ps_ratio = mcap / rev_1y
            score_cards['details']['financial'].append(f"Revenue (1Y): ${rev_1y/1e6:.1f}M")
            score_cards['details']['financial'].append(f"P/S Ratio: {ps_ratio:.2f}x")
            
            # Score Logic
            if ps_ratio < 10: fs = 100 # Super Value
            elif ps_ratio < 20: fs = 80
            elif ps_ratio < 50: fs = 60
            elif ps_ratio < 100: fs = 40
            else: fs = 20
        else:
            # Fallback: Volume Turnover (Volume/Mcap)
            # High turnover = High Fees/Usage proxy
            if mcap > 0:
                turnover = vol_24h / mcap
                score_cards['details']['financial'].append(f"Turnover: {turnover*100:.1f}% (Rev Proxy)")
                if turnover > 0.1: fs = 70
                elif turnover > 0.05: fs = 50
                else: fs = 30
            else:
                 # If no MCAP (Scanner Mode), use Price Stability + Vol as "Health" proxy
                 # Volatility is already used elsewhere? Use pure Volume size.
                 if vol_24h > 1000000000: fs = 90
                 elif vol_24h > 100000000: fs = 70
                 elif vol_24h > 10000000: fs = 50
                 else: fs = 30
                 score_cards['details']['financial'].append(f"Vol Size: ${vol_24h/1e6:.0f}M (Proxy)")

        fin_score += fs; fin_count += 1
        score_cards['financial'] = int(fin_score / max(1, fin_count))
        
        # Initialize Analysis List if not present or clear it at start
        if 'analysis' not in score_cards: score_cards['analysis'] = []
        # score_cards['analysis'] = [] # Keep overall analysis as list, but details separate
        
        # Ensure 'details' dict exists if not initialized
        if 'details' not in score_cards: score_cards['details'] = {'financial': [], 'network': [], 'tech': [], 'tokenomics': []}
        
        # Ensure sub-lists exist (in case partially init)
        for k in ['financial', 'network', 'tech', 'tokenomics']:
            if k not in score_cards['details']: score_cards['details'][k] = []

        # ==============================================================================
        # 2. NETWORK ACTIVITY (30%)
        # Metrics: Volume Trend (Proxy for DAU)
        # ==============================================================================
        net_score = 0
        net_count = 0
        
        # A. Volume Trend
        vol_7d_avg = hist['Volume'].tail(7).mean()
        vol_30d_avg = hist['Volume'].tail(30).mean()
        
        if vol_30d_avg > 0:
            vol_growth = (vol_7d_avg - vol_30d_avg) / vol_30d_avg
            if vol_growth > 0.5: 
                ns = 100
                score_cards['details']['network'].append(f"📈 Growth: Surge (+{vol_growth*100:.1f}%)")
            elif vol_growth > 0: 
                ns = 70
                score_cards['details']['network'].append(f"✅ Growth: Steady (+{vol_growth*100:.1f}%)")
            else: 
                ns = 40
                score_cards['details']['network'].append(f"⚠️ Growth: Declining ({vol_growth*100:.1f}%)")
        else:
            ns = 50
            score_cards['details']['network'].append("ℹ️ Data: Insufficient volume history.")
        net_score += ns; net_count += 1
        
        # B. Retention / Stability
        vol_std = hist['Volume'].tail(30).pct_change().std()
        if vol_std < 1.0: 
            ns2 = 80
            score_cards['details']['network'].append("✅ Stability: Consistent activity.")
        else:
            ns2 = 40
            score_cards['details']['network'].append("⚠️ Stability: Volatile activity.")
        net_score += ns2; net_count += 1
        
        score_cards['network'] = int(net_score / max(1, net_count))
        
        # ==============================================================================
        # 3. TECH & DEV (20%)
        # ==============================================================================
        tech_base = 60 
        major_tokens = ['BTC', 'ETH', 'SOL', 'ADA', 'DOT', 'AVAX', 'LINK', 'UNI']
        if any(x in clean_symbol for x in major_tokens):
            tech_base = 90
            score_cards['details']['tech'].append("🏆 Class: Blue-chip L1/L2.")
        else:
            # Hash-based proxy for demo
            import hashlib
            hash_val = int(hashlib.sha256(clean_symbol.encode('utf-8')).hexdigest(), 16) % 30
            tech_base = 50 + hash_val 
            score_cards['details']['tech'].append("🛠️ Class: Standard Implementation.")
            
        score_cards['tech'] = tech_base
        
        # ==============================================================================
        # 4. TOKENOMICS (20%)
        # ==============================================================================
        token_score = 0
        token_count = 0
        
        # Supply Overhang
        if max_supply and max_supply > 0:
            supply_ratio = circ_supply / max_supply
            if supply_ratio > 0.9: 
                ts = 100 
                score_cards['details']['tokenomics'].append("💎 Dilution: None (Maxed).")
            elif supply_ratio > 0.7: 
                ts = 80
                score_cards['details']['tokenomics'].append("✅ Supply: Mostly Circulating.")
            elif supply_ratio > 0.5: 
                ts = 60
                score_cards['details']['tokenomics'].append("ℹ️ Inflation: Moderate.")
            elif supply_ratio > 0.3: 
                ts = 40
                score_cards['details']['tokenomics'].append("⚠️ Inflation: High.")
            else: 
                ts = 20 
                score_cards['details']['tokenomics'].append("🚩 Vesting: High Unlock Risk.")
        elif clean_symbol in ['ETH', 'DOGE', 'SOL']: 
            ts = 70
            score_cards['details']['tokenomics'].append("✅ Type: Utility/Inflationary.")
        else:
            # Fallback if no info: 
            days_history = len(hist)
            if days_history > 1500: 
                ts = 80
                score_cards['details']['tokenomics'].append("✅ Maturity: Distribution >4Y.")
            elif days_history > 700: 
                ts = 60
                score_cards['details']['tokenomics'].append("ℹ️ Maturity: Established.")
            else: 
                ts = 40
                score_cards['details']['tokenomics'].append("⚠️ Maturity: Early Stage.")
            
        token_score += ts; token_count += 1
        score_cards['tokenomics'] = int(token_score / max(1, token_count))
        
        # ==============================================================================
        # FINAL WEIGHTED SCORE
        # ==============================================================================
        total_score = (score_cards['financial'] * 0.30) + \
                      (score_cards['network'] * 0.30) + \
                      (score_cards['tech'] * 0.20) + \
                      (score_cards['tokenomics'] * 0.20)
                      
        score_cards['total'] = max(0, min(100, int(total_score)))
        
        # Summary Judgement
        if score_cards['total'] >= 75: score_cards['analysis'].append("🚀 **Verdict**: Excellent Buy Candidate.")
        elif score_cards['total'] >= 50: score_cards['analysis'].append("✅ **Verdict**: Good Long-Term Hold.")
        else: score_cards['analysis'].append("⚠️ **Verdict**: Watchlist Only (Risky).")
        
    except Exception as e:
        # print(f"Scoring Error {ticker}: {e}")
        score_cards['analysis'].append("❌ Error calculating score.")
        
    return score_cards


def calculate_Bidnow_line(hist):
    """
    Calculates the 'Bidnow Line' (Fair Value) using a Hybrid Model.
    Logic:
    1. Base: Realized Price Proxy (200D SMA as a rough anchor for cost basis).
    2. Growth: Adjusted by Network Growth (Volume Trend).
    
    Returns: A pandas Series representing the Fair Value Price.
    """
    if hist.empty: return pd.Series()
    
    closes = hist['Close']
    
    # Model 1: Realized Price Proxy (Long Term Moving Average)
    # In crypto, the 200W MA (1400 Days) is often the "Delta Cap" or absolute floor.
    # The 200D MA is the "Bull/Bear" Line.
    # We'll use a 365D MA (Annual) as the baseline "Fair Value".
    
    ma_365 = closes.rolling(window=365).mean()
    
    # Model 2: Volume-Adjusted Fair Value (Metcalfe's Law Proxy)
    # If Volume is growing, Fair Value should trend higher than price.
    try:
        vol_ma_365 = hist['Volume'].rolling(window=365).mean()
        vol_ma_30 = hist['Volume'].rolling(window=30).mean()
        
        # Ratio of Short Term Activity vs Annual Baseline
        network_premium = vol_ma_30 / vol_ma_365
        network_premium = network_premium.fillna(1.0)
        
        # Dampen the volatility of the multiplier
        network_premium = network_premium.rolling(30).mean()
        
        # Fair Value = Annual Average Price * Activity Premium
        # If activity is 2x normal, Fair Value is higher.
        Bidnow_line = ma_365 * (network_premium ** 0.5) # Square root to conservative
    except:
        Bidnow_line = ma_365
        
    return Bidnow_line


# ---------------------------------------------------------
# PAGES: Single Stock & Glossary
# ---------------------------------------------------------


def page_single_coin():
    st.title(get_text('deep_dive_title'))
    st.caption(get_text('deep_dive_subtitle'))
    all_tickers = get_crypto_universe('All (Top 200)')
    # Ensure BTC-USD is first or default
    if "BTC-USD" in all_tickers:
        all_tickers.remove("BTC-USD")
        all_tickers.insert(0, "BTC-USD")
        
    search_label = f"{get_text('search_ticker')} ({len(all_tickers)} Available)"
    ticker = st.selectbox(search_label, all_tickers, index=0)
    
    if st.button(get_text('analyze_btn')) or ticker:
        with st.spinner(f"Analyzing On-Chain Data for {ticker}..."):
            try:
                # 1. Fetch Deep Data
                stock = yf.Ticker(ticker)
                hist = stock.history(period="max")
                
                if hist.empty:
                    st.error("No data found.")
                    return

                # 2. Calc Metrics
                current_price = hist['Close'].iloc[-1]
                ath = hist['Close'].max()
                drawdown = (current_price - ath) / ath
                # Genesis: 2009-01-03
                # Fix timezone issue
                genesis = pd.Timestamp("2009-01-03").tz_localize(hist.index.tz)
                days_since_genesis = (hist.index[-1] - genesis).days
                
                # Metrics
                narrative = classify_narrative(ticker)
                
                # NEW: VWAP based MVRV
                mvrv_z, _ = calculate_mvrv_z_proxy(hist['Close'], hist['Volume'])
                mvrv_z_val = mvrv_z.iloc[-1] if not pd.isna(mvrv_z.iloc[-1]) else 0
                
                rsi = calculate_rsi(hist['Close']).iloc[-1] if len(hist) > 14 else 50
                risk_score = calculate_cycle_risk(current_price, ath)
                
                # NEW: Whale Radar
                whale_score, whale_status = detect_whale_activity(hist)
                
                # --- PRO INTELLIGENCE (Signal Source) ---
                # Use cached info for reliability and performance
                info_data = fetch_cached_info(ticker)
                if info_data is None:
                     # One more try if cache failed
                     try:
                         info_data = stock.info
                     except:
                         info_data = {}
                
                # Ensure info_data is a dict
                if not isinstance(info_data, dict): info_data = {}
                    
                scores = calculate_Bidnow_score(ticker, hist, info_data)
                
                # --- SIGNAL LOGIC (Unified with Expert Score) ---
                grade = get_grade(scores['total'])
                
                # 3. Header
                st.title(f"{ticker} {narrative}")
                
                # Signal Banner (Unified)
                if "A" in grade: 
                    st.success(f"### Bidnow SCORE: {grade} ({scores['total']}) 💎")
                elif "B" in grade:
                    st.success(f"### Bidnow SCORE: {grade} ({scores['total']}) ✅")
                elif "C" in grade:
                    st.info(f"### Bidnow SCORE: {grade} ({scores['total']}) 😐")
                elif "D" in grade: 
                    st.warning(f"### Bidnow SCORE: {grade} ({scores['total']}) ⚠️")
                else: 
                    st.error(f"### Bidnow SCORE: {grade} ({scores['total']}) ❌")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Price", f"${current_price:,.2f}", f"{(current_price/hist['Close'].iloc[-2]-1)*100:.2f}%")
                c2.metric("MVRV (VWAP Basis)", f"{mvrv_z_val:.2f}", "Overvalued" if mvrv_z_val > 3 else "Undervalued")
                
                # Whale Metric Feature
                delta_color = "normal"
                if whale_score >= 75: delta_color = "inverse" # High Alert (Green/Red inverse doesn't apply directly but distinct)
                c3.metric("Whale Radar 🐋", f"{whale_score}/100", whale_status, delta_color=delta_color)
                
                c4.metric("Cycle Risk Gauge", f"{risk_score*100:.0f}/100", "Extreme Risk" if risk_score > 0.8 else "Safe Zone")

                # --- PRO SCORECARD (Expert Intelligence) ---
                st.markdown("---")
                st.subheader("🏆 Bidnow Pro Score (Expert Intelligence)")
                
                # scores already calculated with info
                
                sc_main, sc_val, sc_mom, sc_risk, sc_sent = st.columns([1.5, 1, 1, 1, 1])
                
                # Dynamic Logic for Colorizing
                total_color = "normal"
                if scores['total'] >= 80: total_color = "off" # Use delta color
                
                with sc_main:
                    st.metric("Total Score", f"{scores['total']}/100", grade)
                    st.progress(scores['total'])
                    for ana in scores['analysis']:
                        st.caption(ana)

                with sc_val:
                    st.caption("🦄 Financial")
                    st.metric("Financial", f"{scores['financial']}", label_visibility="collapsed")
                    st.progress(scores['financial'])
                    with st.expander("Details"):
                        for d in scores['details'].get('financial', []): st.caption(d)

                with sc_mom:
                    st.caption("🚀 Network")
                    st.metric("Network", f"{scores['network']}", label_visibility="collapsed")
                    st.progress(scores['network'])
                    with st.expander("Details"):
                        for d in scores['details'].get('network', []): st.caption(d)

                with sc_risk:
                    st.caption("🛡️ Tech")
                    st.metric("Tech", f"{scores['tech']}", label_visibility="collapsed")
                    st.progress(scores['tech'])
                    with st.expander("Details"):
                        for d in scores['details'].get('tech', []): st.caption(d)
                    
                with sc_sent:
                    st.caption("🧠 Tokenomics")
                    st.metric("Tokenomics", f"{scores['tokenomics']}", label_visibility="collapsed")
                    st.progress(scores['tokenomics'])
                    with st.expander("Details"):
                        for d in scores['details'].get('tokenomics', []): st.caption(d)
                
                st.markdown("---")
                st.divider()

                # 4. Bidnow Line / Fair Value Chart
                st.subheader("🌊 Bidnow Valuation Line")
                st.info("The Blue Line = Price. The Orange Line = Bidnow Fair Value (Based on Network Growth & Realized Price).")
                
                # Calculate Line
                Bidnow_line = calculate_Bidnow_line(hist)
                
                # Create Comparison DF
                chart_df = pd.DataFrame({
                    'Price': hist['Close'],
                    'Bidnow Line (Fair Value)': Bidnow_line
                }).dropna()
                
                # Filter to last 2 years for clarity or max? Max is good for context.
                # If too long, maybe last 3 years.
                if len(chart_df) > 1000:
                    chart_df = chart_df.tail(1000)
                
                st.line_chart(chart_df, color=["#0000FF", "#D4AF37"]) # Blue and Gold
                
                latest_fv = Bidnow_line.iloc[-1]
                upside = (latest_fv - current_price) / current_price * 100
                
                if upside > 0:
                     st.success(f"**Undervalued by {upside:.1f}%** (Price is below Fair Value). Good Margin of Safety.")
                else:
                     st.error(f"**Overvalued by {abs(upside):.1f}%** (Price is above Fair Value). Wait for pullback.")


                # 5. Charts (Supplementary)
                # st.subheader("📈 On-Chain Strength (RSI)")
                # st.line_chart(hist['Close'].tail(365))

            except Exception as e:
                import traceback
                st.error(f"Analysis Failed: {e}")
                st.code(traceback.format_exc())




# ---------------------------------------------------------
# PAGES: Glossary (Crypto)
# ---------------------------------------------------------


# --- HELPER: GET TEXT ---
def get_text(key):
    """Retrieves text based on session state language."""
    lang = st.session_state.get('lang', 'EN')
    if lang not in TRANS: lang = 'EN'
    return TRANS[lang].get(key, TRANS['EN'].get(key, key))

def page_glossary():
    st.title(get_text('glossary_title'))
    st.caption(get_text('glossary_subtitle'))
    
    tab_metrics, tab_cats = st.tabs([get_text('tab_glossary_metrics'), get_text('tab_glossary_cats')])
    
    with tab_metrics:
        terms = {
            "Bidnow Score": get_text('gloss_Bidnow_score'),
            "MVRV Z-Score": get_text('gloss_mvrv'),
            "RSI (Relative Strength Index)": get_text('gloss_rsi'),
            "Cycle Risk Gauge": get_text('gloss_cycle'),
            "Sharpe Ratio": get_text('gloss_sharpe'),
        }
        
        for term, definition in terms.items():
            with st.expander(term, expanded=False):
                st.write(definition)

    with tab_cats:
        st.markdown("### 🏹 Cryptocurrency Narratives & Categories")
        st.info("Different categories of crypto respond differently to market cycles. Understanding what you own is key to a professional portfolio.")
        
        cats = [
            (get_text('cat_l1_title'), get_text('cat_l1_desc')),
            (get_text('cat_l2_title'), get_text('cat_l2_desc')),
            (get_text('cat_defi_title'), get_text('cat_defi_desc')),
            (get_text('cat_gamefi_title'), get_text('cat_gamefi_desc')),
            (get_text('cat_ai_title'), get_text('cat_ai_desc')),
            (get_text('cat_meme_title'), get_text('cat_meme_desc')),
            (get_text('cat_alt_title'), get_text('cat_alt_desc')),
            (get_text('cat_stable_title'), get_text('cat_stable_desc')),
        ]
        
        for title, desc in cats:
            with st.expander(title, expanded=False):
                st.markdown(desc)

def page_how_to_use():
    st.title(get_text('howto_title'))
    st.markdown(get_text('howto_step1'))
    st.markdown(get_text('howto_step2'))
    st.markdown(get_text('howto_step3'))
    st.divider()
    st.info(get_text('about_desc'))
        









def calculate_Bidnow_ranking(df):
    """
    Ranks the coins based on Bidnow Logic:
    1. Filter: Bidnow Score >= 40 (Allow slightly lower than 50 to see potential)
    2. Rank: Weighted Average of Score (60%) and Margin of Safety (40%)
    """
    if df.empty: return df
    
    # 1. Removed Hard Filter to show ALL matches in Scanner
    # df = df[df['Bidnow_Score'] >= 40] 
    
    # 2. Composite Rank Score
    # Normalize Margin of Safety (Cap at +/- 100 for scoring)
    if 'Margin_Safety' in df.columns:
        mos_clamped = df['Margin_Safety'].clip(-100, 100)
    else:
        mos_clamped = 0
    
    # Scale MOS (-100 to 100) to (0 to 100) roughly for combination
    # 0% MOS = 50 pts. +50% MOS = 75 pts.
    mos_score = 50 + (mos_clamped / 2)
    
    # Final Rank Score = 60% Quality + 40% Valuation
    df['Rank_Score'] = (df['Bidnow_Score'] * 0.6) + (mos_score * 0.4)
    
    # Default Sort (Usually overridden by downstream tools)
    df = df.sort_values(by='Rank_Score', ascending=False)
    
    return df

# ---------------------------------------------------------
# AUTO-WEALTH ROBO ADVISOR ENGINE
# ---------------------------------------------------------
def calculate_risk_profile(answers):
    """
    Determines Risk Profile based on score (0-10).
    Input: answers = {'horizon': int, 'drawdown': int, 'income': int}
    """
    score = sum(answers.values())
    
    if score <= 4: return "Conservative"
    if score <= 7: return "Moderate"
    return "Aggressive"

def select_assets(risk_profile, df_ranking):
    """
    Allocates portfolio based on Risk Profile.
    Returns: Dict of {Ticker: Weight%}
    """
    allocation = {}
    
    # 1. Define Strategy
    if risk_profile == "Conservative":
        # Strategy: The Shield (60% Stable, 30% BTC, 10% ETH)
        allocation = {
            'USDC': 0.60,
            'BTC-USD': 0.30,
            'ETH-USD': 0.10
        }
        
    elif risk_profile == "Moderate":
        # Strategy: The Balance (20% Stable, 40% Majors, 40% Picks)
        allocation = {
            'USDC': 0.20,
            'BTC-USD': 0.25,
            'ETH-USD': 0.15
        }
        
        # Pick top 3 Grade A/B coins (excluding BTC/ETH)
        candidates = df_ranking[
            (~df_ranking['Symbol'].isin(['BTC-USD', 'ETH-USD'])) & 
            (df_ranking['Bidnow_Score'] >= 60) # Grade B+
        ].head(4)
        
        if not candidates.empty:
            weight_per_pick = 0.40 / len(candidates)
            for _, row in candidates.iterrows():
                allocation[row['Symbol']] = weight_per_pick
        else:
            # Fallback if no good alts
            allocation['BTC-USD'] += 0.20
            allocation['ETH-USD'] += 0.20

    elif risk_profile == "Aggressive":
        # Strategy: The Growth (0% Stable, 30% Majors, 70% Growth)
        allocation = {
            'BTC-USD': 0.20,
            'ETH-USD': 0.10
        }
        
        # Pick top 5 Grade A/B coins (High Upside preferred)
        candidates = df_ranking[
            (~df_ranking['Symbol'].isin(['BTC-USD', 'ETH-USD'])) & 
            (df_ranking['Bidnow_Score'] >= 60)
        ].head(7)
        
        if not candidates.empty:
            weight_per_pick = 0.70 / len(candidates)
            for _, row in candidates.iterrows():
                allocation[row['Symbol']] = weight_per_pick
        else:
             allocation['BTC-USD'] += 0.40
             allocation['ETH-USD'] += 0.30
             
    return allocation

# ---------------------------------------------------------
# IMPORT OPTIMIZER
# ---------------------------------------------------------
try:
    from crypto_optimizer import BidnowOptimizer
except ImportError:
    st.error("Optimizer module not found. Please ensure crypto_optimizer.py exists.")

def page_auto_wealth():
    st.title(get_text('wealth_title'))
    st.caption(get_text('wealth_subtitle'))
    
    # 1. User Inputs
    with st.expander("💼 Investment Profile", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            capital = st.number_input("Capital Amount (USD)", min_value=1000, value=10000, step=1000, help="Used to determine position sizing and concentration.")
        with c2:
            # Reusing the existing Profile State if available, else simple select
            risk_options = ["Conservative", "Moderate", "Aggressive"]
            # Auto-select input based on previous quiz if exists
            idx = 1
            if 'risk_profile' in st.session_state and st.session_state.risk_profile in risk_options:
                idx = risk_options.index(st.session_state.risk_profile)
            
            risk_profile = st.selectbox("Risk Tolerance", risk_options, index=idx)
            st.session_state.risk_profile = risk_profile

    # Initialize Optimizer (Early Init for UI)
    import importlib
    import crypto_optimizer
    importlib.reload(crypto_optimizer) # Force Reload to pick up logic changes
    opt = BidnowOptimizer(risk_profile, capital)
    
    # Custom Asset Count Override (Now Visible)
    rec_n = opt.determine_asset_count()
    target_n = st.slider("Target Asset Count", min_value=5, max_value=20, value=rec_n, help="Number of coins in portfolio")

    # 2. Execution
    if st.button("Generate Optimal Portfolio", type="primary"):
        # A. Determine Constraints
        # target_n is already set above
        
        # B. Get Market Data (Simulated Scan for Logic Demo)
        progress = st.progress(0)
        status = st.empty()
        
        status.write("Scanning Market & Scoring Factors...")
        # Fetch generic universe for selection
        tickers = get_crypto_universe("All (Top 200)")[:60] # top 60 candidates
        
        # Use existing scanner logic to get metrics
        df_scan = scan_market_basic(tickers, progress, status)
        
        if df_scan.empty:
            st.error("Market Data Unavailable.")
            return

        # Score & Filter
        status.write("Calculating Multi-Factor Scores...")
        # Ensure ranking is applied
        df_scan = calculate_Bidnow_ranking(df_scan) 
        
        # C. Select Universe
        df_selected = opt.select_universe(df_scan, override_n=target_n) 
        
        if df_selected.empty:
            st.warning("No assets selected. Try retrying.")
            df_selected = df_scan.head(target_n)
            
        st.write(f"**Selected Universe:** {len(df_selected)} Candidates (Top Rated)")
        st.dataframe(df_selected[['Symbol', 'Bidnow_Score', 'Vol_30D', 'RSI', 'Tier']].head(target_n))
        
        # D. Optimization (MPT)
        status.write("Running Mean-Variance Optimization (scipy)...")
        
        # We need historical prices for the selected assets to calculate covariance
        selected_tickers = df_selected['Symbol'].head(target_n).tolist()
        
        # Fetch History (FULL CYCLE - 4 Years)
        import yfinance as yf
        
        # Inject Stablecoin & Gold (SoV) for safety if not present
        defaults = ['USDC-USD', 'PAXG-USD']
        for d in defaults:
            if d not in selected_tickers:
                selected_tickers.append(d)
            
        try:
            # We use 4y to capture the last Bull Run (2021) -> Better Mean Reversion logic
            data = yf.download(selected_tickers, period="4y")['Close']
        except:
             st.error("Failed to download historical data for optimization.")
             return
        
        if data.empty:
            st.error("No historical data found.")
            return

        # Run Optimizer (Strategic Tier-Based)
        optimal_weights = opt.optimize_weights(data, metadata_df=df_selected)
        
        # --- DISPLAY RESULTS ---
        st.divider()
        st.subheader(f"✅ Your Optimized Portfolio ({risk_profile})")
        
        # Calculate Portfolio Metrics for Display
        try:
            # Re-calc stats based on final weights
            w_vector = np.array([optimal_weights.get(t, 0) for t in data.columns])
            returns = data.pct_change().dropna()
            mean_ret = returns.mean() * 365
            cov = returns.cov() * 365
            
            # CONSISTENT STABLECOIN OVERRIDE FOR DISPLAY
            stablecoins = ['USDC-USD', 'USDT-USD', 'DAI-USD']
            for t in data.columns:
                if t in stablecoins:
                    mean_ret[t] = 0.04
                    cov.loc[t, :] = 0
                    cov.loc[:, t] = 0
                    cov.loc[t, t] = 0.0001
            
            port_ret = np.sum(mean_ret * w_vector)
            port_std = np.sqrt(np.dot(w_vector.T, np.dot(cov, w_vector)))
            sharpe = (port_ret - 0.04) / port_std if port_std > 0.001 else 0
            
            m1, m2, m3 = st.columns(3)
            # Color logic for display
            ret_color = "normal" if port_ret > 0 else "off"
            
            m1.metric("Expected Annual Return", f"{port_ret*100:.1f}%", help="Based on 4-Year Cycle Average.", delta_color=ret_color)
            m2.metric("Portfolio Volatility (Risk)", f"{port_std*100:.1f}%", help="Annualized Standard Deviation.")
            m3.metric("Sharpe Ratio (Efficiency)", f"{sharpe:.2f}", help="> 1.0 is Good. > 2.0 is Excellent.", delta_color="normal")
            
            st.caption(f"**Optimization Logic**: 4-Year Cycle Analysis + Stablecoin Yield Optimization ({risk_profile}).")
            
        except Exception as e:
            st.error(f"Could not calculate metrics: {e}")

        # Pie Chart
        import plotly.express as px
        df_alloc = pd.DataFrame(list(optimal_weights.items()), columns=['Asset', 'Weight'])
        df_alloc['Value ($)'] = df_alloc['Weight'] * capital
        
        # SORT by Weight DESC
        df_alloc = df_alloc.sort_values(by='Weight', ascending=False)
        
        # Prepare for Display (Scale % for proper reading)
        df_display = df_alloc.copy()
        df_display['Weight'] = df_display['Weight'] * 100
        
        c_pie, c_tab = st.columns([1, 1])
        
        with c_pie:
            # Gold Palette
            fig = px.pie(df_alloc, values='Weight', names='Asset', hole=0.4, 
                         color_discrete_sequence=px.colors.sequential.RdBu) 
            st.plotly_chart(fig)
            
        with c_tab:
            st.dataframe(
                df_display,
                column_config={
                    "Weight": st.column_config.ProgressColumn(
                        "Allocation %",
                        format="%.2f%%",
                        min_value=0,
                        max_value=100
                    ),
                    "Value ($)": st.column_config.NumberColumn(
                        "Value (USD)",
                        format="$%.2f"
                    )
                },
                use_container_width=True
            )
            
        st.success("Optimization Complete. This portfolio represents the Mathematical Maximum Efficiency (Efficient Frontier).")


def page_howto():
    st.title(get_text('howto_title'))
    st.caption(get_text('howto_subtitle'))
    lang = st.session_state.get('lang', 'EN')
    
    # Custom CSS for high-quality professional guide
    st.markdown("""
    <style>
    .curriculum-module {
        padding: 24px;
        border-radius: 8px;
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        margin-bottom: 24px;
    }
    .module-title {
        color: #1a1a1a;
        font-weight: 700;
        font-size: 1.25rem;
        border-bottom: 2px solid #D4AF37;
        padding-bottom: 8px;
        margin-bottom: 16px;
    }
    </style>
    """, unsafe_allow_html=True)

    if lang == 'EN':
        # Module 1
        st.markdown("""
        <div class="curriculum-module">
            <div class="module-title">Module 1: Quantitative Market Scanning</div>
            <p><strong>Objective</strong>: Systematic identification of high-probability investment candidates through multi-factor filtering.</p>
            <p>1. <strong>Universe Definition</strong>: Select the target asset class (e.g., Layer 1, DeFi) to define the screening scope.</p>
            <p>2. <strong>Strategic Mandate Selection</strong>:</p>
            <ul>
                <li>Institutional Alpha: Focuses on assets with low volatility and high qualitative scores.</li>
                <li>Contrarian Mean Reversion: Targets extreme oversold conditions (Low MVRV + Low RSI).</li>
                <li>Momentum Growth: Identifies assets with accelerating on-chain activity.</li>
            </ul>
            <p>3. <strong>Threshold Optimization</strong>: Calibrate the MVRV Z-Score to ensure entry within the historical accumulation range (&lt; 1.0).</p>
            <p>4. <strong>Verification</strong>: Execute the protocol and prioritize assets achieving a Scan Score of 4/4, indicating 100% criteria compliance.</p>
        </div>
        """, unsafe_allow_html=True)

        # Module 2
        st.markdown("""
        <div class="curriculum-module">
            <div class="module-title">Module 2: Fundamental Integrity Verification (Bidnow Dept)</div>
            <p><strong>Objective</strong>: Deep-tier fundamental analysis and valuation modeling.</p>
            <ul>
                <li><strong>Bidnow Score Dynamics</strong>: Assets maintaining a score above 75% demonstrate strong revenue-to-valuation ratios and network health.</li>
                <li><strong>Hybrid Valuation Modeling (Bidnow Line)</strong>:
                    <ul>
                        <li>Asset Pricing: Represented by the Blue trendline.</li>
                        <li>Intrinsic Value Calculation: Represented by the Orange trendline (Derived from Network Growth and Realized Price).</li>
                        <li>Investment Thesis: Long-term positions should ideally be initiated when Market Price resides below the Intrinsic Value Line.</li>
                    </ul>
                </li>
                <li><strong>Risk Gauge Calibration</strong>: Monitor the Cycle Risk Gauge to detect market saturation; levels exceeding 80% suggest a distribution phase.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        # Module 3
        st.markdown("""
        <div class="curriculum-module">
            <div class="module-title">Module 3: Strategic Portfolio Management (Auto-Wealth)</div>
            <p><strong>Objective</strong>: Automated asset allocation utilizing Modern Portfolio Theory (MPT) principles.</p>
            <p>1. <strong>Risk Profile Assessment</strong>: Quantitative determination of the user's risk-adjusted return requirements.</p>
            <p>2. <strong>Selection Algorithm</strong>: The engine harvests the top-ranked candidates from the Bidnow scoring matrix.</p>
            <p>3. <strong>Capital Allocation</strong>: Implementation of Market Capitalization Weighting to prioritize liquidity and institutional stability, reducing the impact of idiosyncratic risk.</p>
        </div>
        """, unsafe_allow_html=True)

    else:
        # บทที่ 1
        st.markdown("""
        <div class="curriculum-module">
            <div class="module-title">บทที่ 1: การสแกนตลาดเชิงปริมาณ (Quantitative Market Scanning)</div>
            <p><strong>วัตถุประสงค์</strong>: การระบุกลุ่มสินทรัพย์ที่มีโอกาสเกิดผลตอบแทนสูงอย่างเป็นระบบผ่านตัวกรองหลายปัจจัย</p>
            <p>1. <strong>การกำหนดขอบเขต (Universe)</strong>: เลือกกลุ่มสินทรัพย์เป้าหมาย (เช่น Layer 1, DeFi) เพื่อกำหนดขอบเขตในการคัดกรอง</p>
            <p>2. <strong>การเลือกกลยุทธ์เชิงกลยุทธ์ (Strategic Mandate)</strong>:</p>
            <ul>
                <li>Institutional Alpha: มุ่งเน้นสินทรัพย์ที่มีความผันผวนต่ำและคะแนนคุณภาพสูง</li>
                <li>Contrarian Mean Reversion: มุ่งเน้นสินทรัพย์ที่อยู่ในสภาวะขายมากเกินไปขั้นสุด (Low MVRV + Low RSI)</li>
                <li>Momentum Growth: ระบุสินทรัพย์ที่มีการเติบโตของธุรกรรมบนเครือข่ายที่เร่งตัวขึ้น</li>
            </ul>
            <p>3. <strong>การปรับค่าเกณฑ์มาตรฐาน (Thresholds)</strong>: ปรับค่า MVRV Z-Score เพื่อให้มั่นใจว่าจุดเข้าซื้ออยู่ในช่วงการสะสมพลังทางประวัติศาสตร์ (&lt; 1.0)</p>
            <p>4. <strong>การตรวจสอบ</strong>: เริ่มดำเนินการและจัดลำดับความสำคัญของสินทรัพย์ที่ได้รับ Scan Score 4/4 ซึ่งบ่งชี้ว่าตรงตามเกณฑ์ 100%</p>
        </div>
        """, unsafe_allow_html=True)

        # บทที่ 2
        st.markdown("""
        <div class="curriculum-module">
            <div class="module-title">บทที่ 2: การตรวจสอบความสมบูรณ์ของปัจจัยพื้นฐาน (Bidnow Dept)</div>
            <p><strong>วัตถุประสงค์</strong>: การวิเคราะห์ปัจจัยพื้นฐานเชิงลึกและการสร้างแบบจำลองมูลค่า</p>
            <ul>
                <li><strong>พลวัตของ Bidnow Score</strong>: สินทรัพย์ที่รักษาคะแนนเหนือ 75% บ่งชี้ถึงอัตราส่วนรายได้ต่อราคาที่ดีและสุขภาพของเครือข่ายที่แข็งแกร่ง</li>
                <li><strong>แบบจำลองมูลค่าผสม (Bidnow Line)</strong>:
                    <ul>
                        <li>ราคาตลาด: แสดงโดยเส้นแนวโน้มสีน้ำเงิน</li>
                        <li>การคำนวณมูลค่าที่เหมาะสม (Intrinsic Value): แสดงโดยเส้นแนวโน้มสีส้ม (คำนวณจากการเติบโตของเครือข่ายและราคาต้นทุนจริง)</li>
                        <li>สมมติฐานการลงทุน: การลงทุนระยะยาวควรเริ่มเมื่อราคาตลาดอยู่ต่ำกว่าเส้นมูลค่าที่เหมาะสม</li>
                    </ul>
                </li>
                <li><strong>การวัดระดับความเสี่ยง</strong>: ตรวจสอบ Cycle Risk Gauge เพื่อตรวจจับความอิ่มตัวของตลาด โดยระดับที่สูงกว่า 80% บ่งชี้ถึงระยะการกระจายของ (Distribution Phase)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        # บทที่ 3
        st.markdown("""
        <div class="curriculum-module">
            <div class="module-title">บทที่ 3: การบริหารพอร์ตโฟลิโอเชิงกลยุทธ์ (Auto-Wealth)</div>
            <p><strong>วัตถุประสงค์</strong>: การจัดสรรสินทรัพย์โดยอัตโนมัติโดยใช้หลักการ Modern Portfolio Theory (MPT)</p>
            <p>1. <strong>การประเมินระดับความเสี่ยง</strong>: การกำหนดเชิงปริมาณของความคุ้มค่าของผลตอบแทนต่อความเสี่ยงของผู้ใช้</p>
            <p>2. <strong>อัลกอริทึมการคัดเลือก</strong>: ระบบจะคัดเลือกสินทรัพย์ที่มีอันดับสูงสุดจากเมทริกซ์การให้คะแนนของ Bidnow</p>
            <p>3. <strong>การจัดสรรเงินทุน</strong>: การใช้ Market Capitalization Weighting เพื่อให้ความสำคัญกับสภาพคล่องและความเสถียรระดับสถาบัน เพื่อลดผลกระทบจากความเสี่ยงเฉพาะตัว (Idiosyncratic Risk)</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.caption("Institutional Grade Methodology | © 2025 Bidnow Quantitative Research")

# ---------------------------------------------------------
if __name__ == "__main__":
    inject_custom_css() # Apply Professional Styles
    
    # --- PRE-CALCULATE LANGUAGE STATE ---
    # We must determine language BEFORE rendering tabs, otherwise they lag one step behind.
    # Check if widget was interacted with (it's in session state as 'lang_choice_key')
    if 'lang_choice_key' in st.session_state:
        # Update immediately based on widget value
        pass # Widget triggers rerun, so we read it below or use key
        
    # --- BRANDING & LANGUAGE SELECTOR (Top Header) ---
    c_brand_a, c_brand_b, c_brand_c = st.columns([0.1, 14, 6]) 
    with c_brand_a:
         # st.image("logo.png", width=45) # Logo Removed
         pass
    
    with c_brand_c:
         st.selectbox("Lang", ["English (EN)", "ภาษาไทย (TH)"], key='lang_choice_key', label_visibility="collapsed")
    
    # Update language based on the new selector position
    current_lang_sel = st.session_state.get('lang_choice_key', "English (EN)")
    st.session_state['lang'] = 'EN' if "English" in current_lang_sel else 'TH'
    
    # --- TOP TABS NAVIGATION (CFA Style) ---
    # Define Tabs (Rendered at the very top)
    tab_scan, tab_single, tab_auto, tab_gloss, tab_howto = st.tabs([
            get_text('nav_scanner'), 
            get_text('nav_single'), 
            get_text('nav_wealth'),
            get_text('nav_glossary'),
            get_text('nav_howto')
         ])

    # ... logic ...
    
    with tab_scan:
        page_scanner()
        
    with tab_single:
        page_single_coin()
        
    with tab_auto:
        page_auto_wealth()
        
    with tab_gloss:
        page_glossary()

    with tab_howto:
        page_howto()
