import numpy as np
import pandas as pd
from scipy.optimize import minimize

class BidnowOptimizer:
    def __init__(self, risk_profile, capital_amount):
        """
        Initializes the Optimizer.
        :param risk_profile: 'Conservative', 'Moderate', 'Aggressive'
        :param capital_amount: Float, USD capital available.
        """
        self.risk_profile = risk_profile
        self.capital = capital_amount
        
    def determine_asset_count(self):
        """
        Returns optimal number of assets (N) based on Capital and Marginal Utility of Diversification.
        """
        if self.capital < 10000:
            # Accumulation Phase: Concentrate to build wealth (5-7 assets)
            return 6 
        elif self.capital < 100000:
            # Growth Phase: Balance (8-12 assets)
            return 10
        else:
            # Preservation Phase: Diversify (12-18 assets)
            return 15

    def select_universe(self, ranking_df, override_n=None):
        """
        Selects assets using a 'Pyramid' Tiered Strategy.
        Tier 1 (Foundation): Top Trusted (Blue Chips)
        Tier 2 (Growth): High Bidnow Score (Quality)
        Tier 3 (Alpha): High Momentum/Volatility (Potential)
        """
        if ranking_df.empty: return pd.DataFrame()
        
        df = ranking_df.copy()
        
        # Helper: Ensure we have data
        req_cols = ['Symbol', 'Bidnow_Score', 'Vol_30D', 'RSI']
        for c in req_cols:
            if c not in df.columns: return pd.DataFrame()
            
        # --- PYRAMID SELECTION ---
        target_n = override_n if override_n is not None else self.determine_asset_count()
        
        # 1. Foundation (Blue Chips) - Safety
        # We assume known list or top matches
        blue_chips = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'AVAX-USD', 'USDC-USD', 'PAXG-USD']
        foundation_candidates = df[df['Symbol'].isin(blue_chips)].sort_values(by='Bidnow_Score', ascending=False)
        
        # Select Top 3-5 Foundation
        n_foundation = max(2, int(target_n * 0.4)) # 40% count
        selected_foundation = foundation_candidates.head(n_foundation)
        
        # 2. Core Growth (Quality) - High Score
        # Exclude already selected
        remaining = df[~df['Symbol'].isin(selected_foundation['Symbol'])]
        # Sort by Score
        growth_candidates = remaining.sort_values(by='Bidnow_Score', ascending=False)
        
        n_growth = max(3, int(target_n * 0.4)) # 40% count
        selected_growth = growth_candidates.head(n_growth)
        
        # 3. Alpha (Momentum) - High RSI/Vol (but check for reasonable limits)
        remaining = remaining[~remaining['Symbol'].isin(selected_growth['Symbol'])]
        # Sort by RSI (Momentum) or Vol
        alpha_candidates = remaining.sort_values(by='RSI', ascending=False)
        
        n_alpha = max(2, int(target_n * 0.2)) # 20% count
        selected_alpha = alpha_candidates.head(n_alpha)
        
        # Combine
        combined_df = pd.concat([selected_foundation, selected_growth, selected_alpha]).drop_duplicates(subset=['Symbol'])
        
        # Label Tiers
        combined_df['Tier'] = 'Growth'
        combined_df.loc[combined_df['Symbol'].isin(blue_chips), 'Tier'] = 'Foundation'
        combined_df.loc[combined_df['Symbol'].isin(selected_alpha['Symbol']), 'Tier'] = 'Alpha'
        
        return combined_df.head(target_n)

    def optimize_weights(self, price_history_df, metadata_df=None):
        """
        Calculates optimal weights using Modern Portfolio Theory (MVO).
        Objective: Maximize Sharpe Ratio.
        Features:
        - SoV Base Constraint (BTC+Gold >= 50%)
        - Smart Cash (Yield 4%)
        - Minimum Liquidity Buffer (Cash >= 10%)
        """
        if price_history_df.empty: return {}
        
        # 1. Data Prep: Calculate Returns
        # Drop check: Ensure no NaNs
        prices = price_history_df.dropna()
        if len(prices) < 30: return {} 
        
        returns = prices.pct_change().dropna()
        mean_returns = returns.mean() * 365 # Annualized
        cov_matrix = returns.cov() * 365    # Annualized
        
        tickers = returns.columns.tolist()
        num_assets = len(tickers)
        
        # --- STABLECOIN & SOV SETUP ---
        stablecoins = ['USDC-USD', 'USDT-USD', 'DAI-USD', 'USDE-USD', 'FDUSD-USD']
        sov_assets = ['BTC-USD'] # Pure Association: BTC is the only SoV
        
        # Override Params for Stablecoins (The Risk-Free Yield Trick)
        for i, t in enumerate(tickers):
            if t in stablecoins:
                mean_returns[t] = 0.04  # 4% Risk Free Yield
                cov_matrix.loc[t, :] = 0
                cov_matrix.loc[:, t] = 0
                cov_matrix.loc[t, t] = 0.0001
        
        # 2. Define Constraints based on Risk Profile
        sov_target = 0.50 # SoV Base (BTC Only)
        min_cash = 0.10   # Liquidity Buffer
        
        if self.risk_profile == 'Aggressive':
            max_w = 0.60
            sov_target = 0.40
            min_cash = 0.05
            risk_free_rate = 0.03 
        elif self.risk_profile == 'Conservative':
            max_w = 0.30 
            sov_target = 0.60
            min_cash = 0.15
            risk_free_rate = 0.04
        elif self.risk_profile == 'Sniper':
            # High Conviction: Concentrated Bets
            max_w = 0.80      # Can go almost all-in on one winner
            sov_target = 0.30 # Lower safety base
            min_cash = 0.05   # Bare minimum liquid
            risk_free_rate = 0.04
        elif self.risk_profile == 'Moonshot':
            # Degen Mode: Scatter shots for Multi-baggers
            max_w = 0.40      # Don't bet everything on one junk coin
            sov_target = 0.15 # Minimal BTC anchor
            min_cash = 0.02   # Almost fully invested
            risk_free_rate = 0.0
        else: # Moderate
            max_w = 0.40
            sov_target = 0.50
            min_cash = 0.10
            risk_free_rate = 0.035

        # Bounds
        bounds = []
        for t in tickers:
            if t in stablecoins:
                bounds.append((0.0, 0.8)) # Cash can go high
            elif t in sov_assets:
                bounds.append((0.05, 0.8)) # SoV can go high
            else:
                 # Survival of the Fittest: Allow 0% for weak assets
                bounds.append((0.0, max_w)) 
        bounds = tuple(bounds)
        
        # Constraints
        cons_list = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        # C2: SoV Constraint (BTC >= target)
        sov_indices = [i for i, t in enumerate(tickers) if t in sov_assets]
        if sov_indices:
             cons_list.append({'type': 'ineq', 'fun': lambda x: np.sum(x[sov_indices]) - sov_target})
             
        # C3: Maximum Drawdown Protection / Liquidity Constraint
        # Must hold at least min_cash in Stablecoins
        cash_indices = [i for i, t in enumerate(tickers) if t in stablecoins]
        if cash_indices:
             cons_list.append({'type': 'ineq', 'fun': lambda x: np.sum(x[cash_indices]) - min_cash})

        constraints = tuple(cons_list)
        
        # 3. Objective Function (Negative Sharpe)
        def neg_sharpe_ratio(weights):
            weights = np.array(weights)
            portfolio_return = np.sum(mean_returns * weights)
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            # Safety check for div by zero
            if portfolio_std < 1e-6: return 0
            sharpe = (portfolio_return - risk_free_rate) / portfolio_std
            return -sharpe

        # 4. Run Optimization
        init_guess = num_assets * [1. / num_assets]
        
        try:
            # SLSQP is robust for inequality constraints
            result = minimize(neg_sharpe_ratio, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                optimal_weights = result.x
            else:
                # print(f"Opt Failed: {result.message}")
                optimal_weights = init_guess
                
        except Exception as e:
            # print(f"Optimization Error: {e}")
            optimal_weights = init_guess

        # 5. Format Output
        final_weights = {}
        for i, ticker in enumerate(tickers):
            w = optimal_weights[i]
            if w > 0.001: 
                final_weights[ticker] = round(w, 4)
                
        # Handle normalization
        tot = sum(final_weights.values())
        if tot > 0:
            final_weights = {k: round(v/tot, 4) for k, v in final_weights.items()}
            
        return final_weights
