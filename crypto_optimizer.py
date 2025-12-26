import numpy as np
import pandas as pd
from scipy.optimize import minimize

class BitnowOptimizer:
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
        Tier 2 (Growth): High Bitnow Score (Quality)
        Tier 3 (Alpha): High Momentum/Volatility (Potential)
        """
        if ranking_df.empty: return pd.DataFrame()
        
        df = ranking_df.copy()
        
        # Helper: Ensure we have data
        req_cols = ['Symbol', 'Bitnow_Score', 'Vol_30D', 'RSI']
        for c in req_cols:
            if c not in df.columns: return pd.DataFrame()
            
        # --- PYRAMID SELECTION ---
        target_n = override_n if override_n is not None else self.determine_asset_count()
        
        # 1. Foundation (Blue Chips) - Safety
        # We assume known list or top matches
        blue_chips = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'AVAX-USD']
        foundation_candidates = df[df['Symbol'].isin(blue_chips)].sort_values(by='Bitnow_Score', ascending=False)
        
        # Select Top 3-5 Foundation
        n_foundation = max(2, int(target_n * 0.3)) # 30% count
        selected_foundation = foundation_candidates.head(n_foundation)
        
        # 2. Core Growth (Quality) - High Score
        # Exclude already selected
        remaining = df[~df['Symbol'].isin(selected_foundation['Symbol'])]
        # Sort by Score
        growth_candidates = remaining.sort_values(by='Bitnow_Score', ascending=False)
        
        n_growth = max(3, int(target_n * 0.5)) # 50% count
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
        Calculates optimal weights using Strategic Asset Allocation (Tier-Based).
        Emulates a Professional Crypto Fund structure:
        - Foundation (Safe): ~50%
        - Growth (Core): ~30%
        - Alpha (Moon): ~20%
        
        Within tiers, weights are distributed based on 'Bitnow_Score' (Fundamental Quality).
        """
        if price_history_df.empty: return {}
        tickers = price_history_df.columns.tolist()
        num_assets = len(tickers)
        
        # Default fallback
        if metadata_df is None or 'Tier' not in metadata_df.columns:
             return {t: round(1.0/num_assets, 4) for t in tickers}

        # --- STRATEGIC ALLOCATION ---
        weights = {}
        
        # 1. Define Tier Targets based on Risk Profile
        if self.risk_profile == 'Conservative':
            tier_targets = {'Foundation': 0.60, 'Growth': 0.30, 'Alpha': 0.10}
        elif self.risk_profile == 'Aggressive':
            tier_targets = {'Foundation': 0.30, 'Growth': 0.40, 'Alpha': 0.30}
        else: # Moderate
            tier_targets = {'Foundation': 0.50, 'Growth': 0.30, 'Alpha': 0.20}
            
        # 2. Group Assets
        grouped = metadata_df[metadata_df['Symbol'].isin(tickers)].groupby('Tier')
        
        total_assigned = 0
        
        # Calculate sub-weights for each tier
        for tier, target_pct in tier_targets.items():
            # Get assets in this tier
            # Note: Tier names must match what we assigned in select_universe
            # select_universe uses: 'Foundation', 'Growth', 'Alpha'
            
            tier_assets = metadata_df[(metadata_df['Symbol'].isin(tickers)) & (metadata_df['Tier'] == tier)]
            
            if tier_assets.empty:
                # Distribute this tier's target to others or ignore
                continue
                
            # Weighting within Tier: Score Weighted
            # Higher Score = Higher Weight
            total_score = tier_assets['Bitnow_Score'].sum()
            
            if total_score == 0:
                # Equal weight if no scores
                sub_weight = target_pct / len(tier_assets)
                for _, row in tier_assets.iterrows():
                    weights[row['Symbol']] = sub_weight
            else:
                # Proportional to Score
                for _, row in tier_assets.iterrows():
                    share = row['Bitnow_Score'] / total_score
                    weights[row['Symbol']] = target_pct * share
                    
        # 3. Normalize (in case some tiers were empty)
        total_w = sum(weights.values())
        if total_w > 0:
            weights = {k: v / total_w for k, v in weights.items()}
            
        # 4. Fill missing (if any tickers passed but not in metadata)
        # Should not happen given logic, but safety check
        for t in tickers:
            if t not in weights:
                weights[t] = 0.0

        return {k: round(v, 4) for k, v in weights.items()}
