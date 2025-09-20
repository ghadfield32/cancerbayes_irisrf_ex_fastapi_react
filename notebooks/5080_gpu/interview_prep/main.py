"""
NBA Analysis Pipeline - Optimized for Live Coding Assessments
============================================================


# Detailed Questions for set up 

Start by:
working through the datasets to create VORP, EWA, PER, PIE, and PER in basic steps so that we can easily do this ourselves in easy code. Then let's answer the questions below in order while creating easy to use and write functions that can do these. 

Questions:
“Given a CSV dataset, how would you explore and summarize it?”

“Given a DataFrame, how would you handle missing values?”

“How would you detect and address outliers in a dataset?”

“Perform univariate, bivariate, and multivariate analysis on given columns.”

“Given a dataset, how would you normalize or standardize its features?”

“Write a function to compute summary statistics (mean, median, std, etc.) of a column.”

“Given a dataset, how would you identify the type of each variable and choose feature encoding?”

“Given a dataset and a target variable, how would you check for relationships or correlations?”

“Write code to detect missingness patterns and decide how to impute.”

“Describe the full data-analysis pipeline: from loading to insight delivery—then code accordingly.”





interview questions:
1) "Who are the top 5 players by points per game this season?"
2) "Find all players who average more than 25 PPG and 10 RPG"
3) "What's the correlation between minutes played and points scored?"

4) "Calculate shooting efficiency - players with best True Shooting %"
5) "Find the most 'complete' players - top 10 in points + assists + rebounds"
6) "Which team has the most balanced scoring attack?"

7) "Clean this dataset - handle missing values in FG%"
8) "Group players by minutes played tiers and show average stats"

9) "Is there a significant difference in scoring between guards and forwards?"
10) "Find players who are outliers in efficiency"

11) "Who improved the most from last season to this season?"
12) "Calculate Player Impact Estimate (PIE) for top 10 players"


--
13) Who are the top 3 scorers **per team** by **points per-36** (filter to minutes ≥ 15)?

14) Build a simple **composite impact score** per player using standardized per-36 stats:
   `impact = z(points/36) + 0.7*z(assists/36) + 0.7*z(rebounds/36)`. Who are the top 10?

15) Do players **outperform their expected points** for their minutes/assists/rebounds?
   Fit a quick linear model `points ~ minutes + assists + rebounds` and list top 10 **positive residuals**.

16) Which team is **most balanced vs. star-heavy** by scoring?
   Compute **coefficient of variation** (std/mean) of `points` per team and rank.

17) Bucket players into **minutes tiers**: `[10–19, 20–29, 30–40]`.
   What are the mean/median of `points/assists/rebounds` per tier?

18) “Three-above-median” players: per team, who is **above the team median** in **points, assists, and rebounds** simultaneously?

19) Write a reusable helper `top_k(df, by, k, group=None)` and use it to return the **top 2 rebounders per team**.

20) What’s the **team effect** on scoring after controlling for other stats?
   One-hot encode `team`, fit a Ridge `points ~ minutes + assists + rebounds + team_*`, and show team coefficients.

21) Give a quick **bootstrap 95% CI** for **mean points per team** (1,000 resamples). Which teams have clearly higher means?

22) Detect potential **duplicate identity issues**: do we have any duplicate `(player_id, name)` rows? If so, keep the one with the **max minutes**.



"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict
from scipy import stats
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler  # Added for Ridge regression
from pydantic import BaseModel
from omegaconf import OmegaConf

# ============================================================================
# HELPERS
# ============================================================================

def add_per36(df: pd.DataFrame, stat_cols: list[str], minutes_col: str = "minutes") -> pd.DataFrame:
    """
    Compute per-36 values for the given season-level totals.
    - Does NOT fill NaNs; if minutes == 0 -> NaN (surfaces data issues instead of masking them).
    """
    d = df.copy()
    if minutes_col not in d.columns:
        raise KeyError(f"'{minutes_col}' not found on DataFrame needed for per-36.")
    mins = d[minutes_col].astype(float)
    missing_cols = []
    for s in stat_cols:
        if s not in d.columns:
            missing_cols.append(s)
            continue  # don't fabricate; skip silently
        per36_col = f"{s}_per36"
        d[per36_col] = np.where(mins > 0, (d[s].astype(float) * 36.0) / mins, np.nan)
    
    # Explicit warning for missing columns
    if missing_cols:
        print(f"Warning: Skipped missing stat columns for per-36: {missing_cols}")
    return d

def standardize_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Add z-scored versions of columns as 'z_<col>'.
    - No imputation; if std == 0 or col is all-NaN -> resulting z_ col is NaN.
    """
    d = df.copy()
    for c in cols:
        if c not in d.columns:
            continue
        col = d[c].astype(float)
        mu = col.mean(skipna=True)
        sigma = col.std(ddof=0, skipna=True)
        zc = (col - mu) / sigma if sigma and not np.isnan(sigma) and sigma != 0 else np.nan
        d[f"z_{c}"] = zc
    return d

# ============================================================================
# UNIFIED HELPER FUNCTIONS - Eliminates code duplication
# ============================================================================

def assign_minutes_tier(mpg):
    """
    UNIFIED FUNCTION: Single source of truth for minutes tier assignment
    Used by both question_8_minutes_tiers and question_17_minutes_tier_analysis
    """
    if pd.isna(mpg):
        return "Other"
    elif 10 <= mpg < 20:
        return "10-19 MPG"
    elif 20 <= mpg < 30:
        return "20-29 MPG" 
    elif 30 <= mpg <= 40:
        return "30-40 MPG"
    else:
        return "Other"

def calculate_team_scoring_balance(df: pd.DataFrame, group_cols: list = None) -> pd.DataFrame:
    """
    UNIFIED FUNCTION: Single source of truth for team scoring balance calculation
    Used by both question_6_team_balance and question_16_team_scoring_balance
    
    Args:
        df: DataFrame with team and scoring data
        group_cols: Columns to group by (default: ['team', 'season'])
    
    Returns:
        DataFrame with team balance metrics
    """
    if group_cols is None:
        group_cols = ['team', 'season']
    
    # Ensure team column exists
    df = add_team_column(df.copy())
    
    # Calculate team balance metrics
    team_balance = (df.groupby(group_cols)['PTS']
                   .agg(['mean', 'std', 'count'])
                   .reset_index())
    
    # Calculate coefficient of variation (CV = std/mean)
    team_balance['cv'] = (team_balance['std'] / team_balance['mean'].replace(0, np.nan))
    
    # Classify balance type based on median CV
    median_cv = team_balance['cv'].median()
    team_balance['balance_type'] = np.where(
        team_balance['cv'] < median_cv, 
        'Balanced', 
        'Star-Heavy'
    )
    
    return team_balance.sort_values('cv')

def safe_mpg_calculation(df: pd.DataFrame, minutes_col: str = 'minutes', games_col: str = 'games') -> pd.Series:
    """
    UNIFIED FUNCTION: Safe MPG calculation that handles edge cases
    
    Fixes the issue where games=0 creates inf values
    Used by questions that need MPG calculations
    """
    # Replace 0 games with NaN to avoid division by zero -> inf
    safe_games = df[games_col].replace(0, np.nan)
    mpg = df[minutes_col] / safe_games
    return mpg



# ============================================================================
# CORE DATA LOADING - SIMPLIFIED
# ============================================================================
class ColumnSchema(BaseModel):
    y_variable: str
    ordinal: List[str]
    nominal: List[str]
    numerical: List[str]
    id_cols: List[str]

def load_schema(yaml_path: str) -> ColumnSchema:
    """Load schema from YAML file"""
    cfg = OmegaConf.load(yaml_path)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    return ColumnSchema(**cfg_dict)



def load_data(csv_path: str, schema: ColumnSchema, sample: bool = False, sample_size: int = 100) -> pd.DataFrame:
    """
    Streamlined data loading using schema for column types
    No complex branching - direct pandas operations
    """
    # Build dtype mapping from schema
    dtype_map = {}
    
    # ID columns as strings for joining
    for col in schema.id_cols:
        dtype_map[col] = 'string'
    
    # Nominal columns as categories
    for col in schema.nominal:
        dtype_map[col] = 'string'
    
    # Load data
    if sample:
        data = pd.read_csv(csv_path, dtype=dtype_map, nrows=sample_size)
    else:
        data = pd.read_csv(csv_path, dtype=dtype_map)
    
    # Convert numerical columns - let pandas handle the rest
    for col in schema.numerical:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
    
    return data

def normalize_player_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Streamlined player data normalization
    Direct column operations without fallbacks
    """
    d = df.copy()
    
    # Standard column mapping - no conditional logic
    column_mapping = {
        'personId': 'player_id', 'gameId': 'game_id', 'numMinutes': 'minutes',
        'points': 'PTS', 'assists': 'AST', 'reboundsTotal': 'TRB',
        'reboundsOffensive': 'OREB', 'reboundsDefensive': 'DREB',
        'steals': 'STL', 'blocks': 'BLK', 'turnovers': 'TOV',
        'fieldGoalsAttempted': 'FGA', 'fieldGoalsMade': 'FGM',
        'freeThrowsAttempted': 'FTA', 'freeThrowsMade': 'FTM',
        'threePointersAttempted': '3PA', 'threePointersMade': '3PM',
        'foulsPersonal': 'PF'
    }
    
    d = d.rename(columns=column_mapping)
    
    # Create name field directly
    d['name'] = (d['firstName'].astype('string') + ' ' + d['lastName'].astype('string')).str.strip()
    
    # Convert numeric stats - no error handling, let NaNs surface
    numeric_stats = ['minutes', 'PTS', 'AST', 'TRB', 'OREB', 'DREB', 
                    'STL', 'BLK', 'TOV', 'FGA', 'FGM', 'FTA', 'FTM', '3PA', '3PM', 'PF']
    
    for col in numeric_stats:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors='coerce')  # Keep NaNs to surface data issues
    
    return d

def build_season_totals(player_games: pd.DataFrame) -> pd.DataFrame:
    """
    FIXED VERSION: Season aggregation with proper NaN handling
    
    MAIN FIX: Handles KeyError: nan by checking for valid minutes before using idxmax()
    """
    if player_games.empty:
        raise ValueError("Player games DataFrame cannot be empty")
    
    print(f"Building season totals from {len(player_games)} game records")
    
    # FIXED: Team assignment function that handles NaN minutes
    def get_team_assignment(group):
        """Get team assignment, handling NaN minutes properly"""
        # Filter out NaN minutes before finding max
        valid_minutes = group[group['minutes'].notna()]
        
        if valid_minutes.empty:
            # If no valid minutes, use first available record
            print(f"Warning: No valid minutes for {group.iloc[0]['name']} in {group.iloc[0]['season']}")
            return group.iloc[0][['playerteamCity', 'playerteamName']]
        else:
            # Use the game with maximum minutes
            max_idx = valid_minutes['minutes'].idxmax()
            return group.loc[max_idx, ['playerteamCity', 'playerteamName']]
    
    # FIXED: Apply the safe team assignment function
    try:
        team_assignment = (
            player_games.groupby(['player_id', 'name', 'season'])
            .apply(get_team_assignment)
            .reset_index()
        )
        print(f"Team assignments created: {len(team_assignment)} unique player-seasons")
    except Exception as e:
        print(f"Error in team assignment: {e}")
        # Debug info
        print("Sample of problematic data:")
        sample_group = player_games.groupby(['player_id', 'name', 'season']).first().head()
        print(sample_group[['minutes', 'playerteamCity', 'playerteamName']])
        raise
    
    # Aggregate stats with validation
    agg_cols = {
        'minutes': 'sum', 'PTS': 'sum', 'AST': 'sum', 'TRB': 'sum',
        'OREB': 'sum', 'DREB': 'sum', 'STL': 'sum', 'BLK': 'sum',
        'PF': 'sum', 'TOV': 'sum', 'FGA': 'sum', 'FGM': 'sum',
        'FTA': 'sum', 'FTM': 'sum', '3PA': 'sum', '3PM': 'sum',
        'game_id': 'nunique'
    }
    
    # Only aggregate columns that exist
    available_agg_cols = {k: v for k, v in agg_cols.items() if k in player_games.columns}
    missing_agg_cols = [k for k in agg_cols.keys() if k not in player_games.columns]
    
    if missing_agg_cols:
        print(f"Warning: Missing aggregation columns: {missing_agg_cols}")
    
    season_stats = (
        player_games.groupby(['player_id', 'name', 'season'])
        .agg(available_agg_cols)
        .rename(columns={'game_id': 'games'})
        .reset_index()
    )
    
    print(f"Season stats aggregated: {len(season_stats)} player-seasons")
    
    # Merge team info with validation
    result = season_stats.merge(
        team_assignment, 
        on=['player_id', 'name', 'season'], 
        how='left',
        validate='one_to_one'
    )
    
    # Calculate shooting percentages with proper zero handling
    if 'FGM' in result.columns and 'FGA' in result.columns:
        result['FG_pct'] = result['FGM'] / result['FGA'].replace(0, np.nan)
    
    if 'FTM' in result.columns and 'FTA' in result.columns:
        result['FT_pct'] = result['FTM'] / result['FTA'].replace(0, np.nan)
    
    # Calculate True Shooting with validation
    if all(col in result.columns for col in ['PTS', 'FGA', 'FTA']):
        result['TSA'] = result['FGA'] + 0.44 * result['FTA']
        result['TS_pct'] = result['PTS'] / (2.0 * result['TSA']).replace(0, np.nan)
    
    # Calculate per-game stats with validation
    if 'games' in result.columns:
        games_safe = result['games'].replace(0, np.nan)
        
        if 'PTS' in result.columns:
            result['PPG'] = result['PTS'] / games_safe
        if 'AST' in result.columns:
            result['APG'] = result['AST'] / games_safe
        if 'TRB' in result.columns:
            result['RPG'] = result['TRB'] / games_safe
    
    print(f"Season totals completed: {result.shape}")
    return result

# ============================================================================
# ADVANCED METRICS - SIMPLIFIED
# ============================================================================


def compute_pie(player_games: pd.DataFrame) -> pd.DataFrame:
    """
    Compute player share of game 'PIE numerator' using only observed stats.
    No invented baselines; no imputations.
    """
    d = player_games.copy()
    d['pie_num'] = (
        d['PTS'] + d['FGM'] + d['FTM']
        - d['FGA'] - d['FTA']
        + d['DREB'] + 0.5*d['OREB'] + d['AST'] + d['STL'] + 0.5*d['BLK']
        - d['PF'] - d['TOV']
    )

    game_totals = d.groupby('game_id', as_index=False)['pie_num'].sum().rename(columns={'pie_num':'game_pie_total'})
    d = d.merge(game_totals, on='game_id', how='left')
    d['pie'] = d['pie_num'] / d['game_pie_total'].replace(0, np.nan)

    season_pie = (
        d.groupby(['player_id','name','season'], as_index=False)
         .apply(lambda x: pd.Series({'season_PIE': (x['pie']*x['minutes']).sum() / x['minutes'].sum()
                                     if x['minutes'].sum() > 0 else np.nan}))
         .reset_index(drop=True)
    )
    return season_pie

def compute_per(player_games: pd.DataFrame, team_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean PER calculation with proper 3PT handling - no silent misalignment
    """
    # Calculate league stats including 3PT properly
    league_stats = team_df.groupby('season', as_index=False).agg({
        'fieldGoalsMade': 'sum',
        'fieldGoalsAttempted': 'sum',
        'freeThrowsMade': 'sum',
        'freeThrowsAttempted': 'sum',
        'assists': 'sum',
        'reboundsOffensive': 'sum',
        'reboundsTotal': 'sum',
        'turnovers': 'sum',
        'threePointersMade': 'sum'  # Include 3PT in aggregation to avoid misalignment
    })
    
    # Calculate possessions and VOP
    league_stats['team_poss'] = (
        league_stats['fieldGoalsAttempted'] - league_stats['reboundsOffensive'] +
        league_stats['turnovers'] + 0.44 * league_stats['freeThrowsAttempted']
    )
    # FIXED: Include free throws in league points calculation for accurate VOP
    league_stats['lg_pts'] = 2 * league_stats['fieldGoalsMade'] + league_stats['threePointersMade'] + league_stats['freeThrowsMade']
    league_stats['VOP'] = league_stats['lg_pts'] / league_stats['team_poss'].replace(0, np.nan)
    
    # Merge with player data
    d = player_games.merge(league_stats[['season', 'VOP']], on='season', how='left')
    
    # Calculate raw PER components
    d['raw_per'] = (
        d['3PM'] + 0.667 * d['AST'] + d['FGM'] + 0.5 * d['FTM'] +
        d['VOP'] * (d['TRB'] + d['STL'] + d['BLK']) -
        d['VOP'] * (d['TOV'] + (d['FGA'] - d['FGM']) + 0.44 * (d['FTA'] - d['FTM']))
    )
    
    # Season PER - weighted by minutes
    season_per = (
        d.groupby(['player_id', 'name', 'season'], as_index=False)
         .apply(lambda x: pd.Series({
             'season_PER': (x['raw_per'] * x['minutes']).sum() / x['minutes'].sum()
             if x['minutes'].sum() > 0 else np.nan
         }))
         .reset_index(drop=True)
    )
    
    return season_per

def calculate_metrics(player_df: pd.DataFrame, team_df: pd.DataFrame) -> pd.DataFrame:
    """
    ENHANCED VERSION: Main calculation pipeline with comprehensive error handling
    """
    print("=== CALCULATING NBA METRICS (ENHANCED VERSION) ===")
    
    try:
        # Step 1: Validate inputs
        if player_df.empty:
            raise ValueError("Player DataFrame is empty")
        if team_df.empty:
            raise ValueError("Team DataFrame is empty")
        
        print(f"Input validation passed:")
        print(f"  Player records: {len(player_df):,}")
        print(f"  Team records: {len(team_df):,}")
        
        # Step 2: Normalize data with validation
        print("\nStep 1: Normalizing player data...")
        player_normalized = normalize_player_data(player_df)
        print(f"✓ Normalized player data: {player_normalized.shape}")
        
        # Validate normalization
        required_cols = ['player_id', 'name', 'season', 'minutes']
        missing_required = [col for col in required_cols if col not in player_normalized.columns]
        if missing_required:
            raise ValueError(f"Missing required columns after normalization: {missing_required}")

        # Step 3: Build season totals with the FIXED function
        print("\nStep 2: Building season totals...")
        season_totals = build_season_totals(player_normalized)
        print(f"✓ Season totals: {season_totals.shape}")
        
        # Validate season totals
        if season_totals.empty:
            raise RuntimeError("Season totals calculation resulted in empty DataFrame")

        # Step 4: Calculate PIE metrics
        print("\nStep 3: Calculating PIE...")
        try:
            season_pie = compute_pie(player_normalized)
            print(f"✓ PIE calculated: {season_pie.shape}")
        except Exception as e:
            print(f"Warning: PIE calculation failed: {e}")
            # Create empty PIE DataFrame to continue pipeline
            season_pie = pd.DataFrame({
                'player_id': season_totals['player_id'],
                'name': season_totals['name'], 
                'season': season_totals['season'],
                'season_PIE': np.nan
            })

        # Step 5: Calculate PER
        print("\nStep 4: Calculating PER...")
        try:
            per_metrics = compute_per(player_normalized, team_df)
            print(f"✓ PER metrics calculated: {per_metrics.shape}")
        except Exception as e:
            print(f"Warning: PER calculation failed: {e}")
            # Create empty PER DataFrame to continue pipeline
            per_metrics = pd.DataFrame({
                'player_id': season_totals['player_id'],
                'name': season_totals['name'],
                'season': season_totals['season'], 
                'season_PER': np.nan
            })

        # Step 6: Merge everything with validation
        print("\nStep 5: Merging datasets...")
        
        # Merge PIE data
        final_data = season_totals.merge(
            season_pie, 
            on=['player_id', 'name', 'season'], 
            how='left'
        )
        print(f"  After PIE merge: {final_data.shape}")
        
        # Merge PER data  
        final_data = final_data.merge(
            per_metrics, 
            on=['player_id', 'name', 'season'], 
            how='left'
        )
        print(f"  After PER merge: {final_data.shape}")

        # Step 7: Add per-36 stats
        print("\nStep 6: Adding per-36 statistics...")
        per36_stats = ['PTS', 'AST', 'TRB', 'STL', 'BLK', 'TOV']
        available_per36 = [stat for stat in per36_stats if stat in final_data.columns]
        
        if available_per36:
            final_data = add_per36(final_data, stat_cols=available_per36, minutes_col='minutes')
            print(f"✓ Per-36 stats added for: {available_per36}")
        else:
            print("Warning: No stats available for per-36 calculation")

        # Final validation
        print(f"\n✓ Final dataset: {final_data.shape}")
        print(f"✓ Columns: {len(final_data.columns)}")
        
        if final_data.empty:
            raise RuntimeError("Final dataset is empty after processing")
            
        # Quality checks
        total_minutes = final_data['minutes'].sum()
        valid_players = final_data['minutes'].notna().sum()
        print(f"✓ Quality check: {valid_players:,} players with valid minutes totaling {total_minutes:,.0f}")
        
        return final_data
        
    except Exception as e:
        print(f"\n✗ ERROR in calculate_metrics: {str(e)}")
        print(f"✗ Error type: {type(e).__name__}")
        
        # Detailed debugging info
        if 'player_normalized' in locals():
            print(f"\nDebug info:")
            print(f"  Player normalized shape: {player_normalized.shape}")
            print(f"  Player normalized columns: {sorted(player_normalized.columns)}")
            
            # Check for data quality issues
            minutes_issues = player_normalized['minutes'].isna().sum()
            print(f"  Minutes NaN count: {minutes_issues}")
            
            if minutes_issues > 0:
                print(f"  Sample records with NaN minutes:")
                nan_sample = player_normalized[player_normalized['minutes'].isna()].head(3)
                print(nan_sample[['name', 'season', 'minutes', 'PTS']].to_string())
        
        raise


# ============================================================================
# INTERVIEW QUESTION FUNCTIONS - SIMPLIFIED
# ============================================================================

def add_team_column(df: pd.DataFrame) -> pd.DataFrame:
    """Create team column from city and name - handle NaN values properly"""
    if 'team' not in df.columns:
        city = df['playerteamCity'].astype('string').fillna('')
        name = df['playerteamName'].astype('string').fillna('')
        df['team'] = (city + ' ' + name).str.strip()
    return df

def top_k_by_group(df: pd.DataFrame, metric: str, k: int, group_col=None) -> pd.DataFrame:
    """
    IMPROVED VERSION: Handles both single string and list of group columns
    """
    if isinstance(group_col, str):
        return df.groupby(group_col, group_keys=False).apply(lambda x: x.nlargest(k, metric)).reset_index(drop=True)
    elif isinstance(group_col, list):
        return df.groupby(group_col, group_keys=False).apply(lambda x: x.nlargest(k, metric)).reset_index(drop=True)
    else:
        return df.nlargest(k, metric)

# Question implementations

def question_1_top_scorers(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Top K players by PPG - direct operation"""
    return df.nlargest(k, 'PPG')[['name', 'season', 'PPG', 'PTS', 'games']]



def question_2_double_double_threshold(df: pd.DataFrame, pts_threshold: float = 25, reb_threshold: float = 10) -> pd.DataFrame:
    """Players above thresholds in multiple categories"""
    mask = (df['PPG'] >= pts_threshold) & (df['RPG'] >= reb_threshold)
    return df[mask][['name', 'season', 'PPG', 'RPG', 'PTS', 'TRB']]


def question_3_correlation(df: pd.DataFrame, col1: str = 'minutes', col2: str = 'PTS') -> float:
    """Direct correlation calculation without verbose output"""
    correlation = df[col1].corr(df[col2])
    print(f"Correlation between {col1} and {col2}: {correlation:.3f}")
    
    if correlation > 0.7:
        print("This indicates a strong positive relationship")
    elif correlation > 0.3:
        print("This indicates a moderate positive relationship")
        
    return correlation

def question_4_true_shooting_leaders(df: pd.DataFrame) -> pd.DataFrame:
    """Question 4: Calculate shooting efficiency - players with best True Shooting %"""
    print("\n=== Question 4: True Shooting % Leaders ===")
    
    # Step 1: Calculate True Shooting % if not exists
    if 'TS_pct' not in df.columns:
        df['TSA'] = df['FGA'] + 0.44 * df['FTA']
        df['TS_pct'] = df['PTS'] / (2.0 * df['TSA']).replace(0, np.nan)
    
    # Step 2: Filter qualified players (minimum attempts)
    qualified = df[df['TSA'] >= 100].copy()  # At least 100 true shot attempts
    
    # Step 3: Get top 10
    result = qualified.nlargest(10, 'TS_pct')[['name', 'season', 'TS_pct', 'PTS', 'FGA', 'FTA']]
    print(f"Top 10 True Shooting % (min 100 TSA):{result}")
    return result

def question_5_complete_players(df: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """Most complete players by combined stats"""
    df['complete_score'] = df['PTS'] + df['AST'] + df['TRB']
    return df.nlargest(k, 'complete_score')[['name', 'season', 'complete_score', 'PTS', 'AST', 'TRB']]

def question_6_team_balance(df: pd.DataFrame) -> pd.DataFrame:
    """
    UPDATED VERSION: Uses unified calculate_team_scoring_balance function
    """
    print("\n=== Question 6: Team Scoring Balance ===")
    return calculate_team_scoring_balance(df, group_cols=['team', 'season'])

def question_7_clean_fg_percentage(df: pd.DataFrame) -> pd.DataFrame:
    """Clean FG% calculation - keep truthy NaNs instead of fabricating values"""
    df = df.copy()
    df['FG_pct'] = df['FGM'] / df['FGA'].replace(0, np.nan)
    
    missing = int(df['FG_pct'].isna().sum())
    print(f"Missing FG% values: {missing} (left as NaN to preserve data integrity)")
    return df

def question_8_minutes_tiers(df: pd.DataFrame) -> pd.DataFrame:
    """
    FIXED VERSION: Question 8 using unified helper and safe MPG calculation
    
    Fixes: 
    - Uses safe_mpg_calculation to avoid inf values
    - Uses unified assign_minutes_tier function
    """
    print("\n=== Question 8: Minutes Tiers Analysis ===")
    
    # Step 1: Create safe minutes per game calculation
    df = df.copy()
    df['MPG'] = safe_mpg_calculation(df, 'minutes', 'games')
    
    # Step 2: Use unified tier assignment
    df['minutes_tier'] = df['MPG'].apply(assign_minutes_tier)
    
    # Step 3: Calculate tier averages
    tier_stats = (df[df['minutes_tier'] != 'Other']
                  .groupby('minutes_tier')
                  .agg({
                      'PTS': ['mean', 'median'],
                      'AST': ['mean', 'median'],
                      'TRB': ['mean', 'median'],
                      'MPG': ['mean', 'count']
                  }).round(1))
    
    # Flatten column names
    tier_stats.columns = ['_'.join(col).strip() for col in tier_stats.columns]
    tier_stats = tier_stats.reset_index()
    
    print("Average stats by minutes tier:")
    for i, row in tier_stats.iterrows():
        print(f"\n{row['minutes_tier']} (n={row['MPG_count']:.0f}):")
        print(f"  Points: {row['PTS_mean']:.1f} avg, {row['PTS_median']:.1f} median")
        print(f"  Assists: {row['AST_mean']:.1f} avg, {row['AST_median']:.1f} median")
        print(f"  Rebounds: {row['TRB_mean']:.1f} avg, {row['TRB_median']:.1f} median")
    
    return tier_stats

def question_9_guards_vs_forwards_scoring(df: pd.DataFrame) -> Dict:
    """Require a real 'position' column; do not simulate."""
    print("\n=== Question 9: Guards vs Forwards Scoring ===")
    if "position" not in df.columns:
        raise ValueError("This analysis requires a real 'position' column. "
                         "Provide roster/position data or skip this question.")

    guards_mask = df["position"].isin(["PG","SG"])
    fwds_mask = df["position"].isin(["SF","PF","C"])
    guards = df.loc[guards_mask, "PTS"]
    forwards = df.loc[fwds_mask, "PTS"]

    t_stat, p_value = stats.ttest_ind(guards, forwards, equal_var=False, nan_policy="omit")
    results = {
        'guards_mean': guards.mean(), 'guards_std': guards.std(), 'guards_n': guards.shape[0],
        'forwards_mean': forwards.mean(), 'forwards_std': forwards.std(), 'forwards_n': forwards.shape[0],
        't_statistic': t_stat, 'p_value': p_value, 'significant': p_value < 0.05
    }
    print(f"Guards (n={results['guards_n']}): {results['guards_mean']:.1f} ± {results['guards_std']:.1f}")
    print(f"Forwards (n={results['forwards_n']}): {results['forwards_mean']:.1f} ± {results['forwards_std']:.1f}")
    print(f"T-test: t = {results['t_statistic']:.3f}, p = {results['p_value']:.3f} "
          f"=> Significant: {'Yes' if results['significant'] else 'No'}")
    return results


def question_10_efficiency_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    IMPROVED VERSION: Better labeling of efficiency metric when PER not available
    
    Fixes:
    - Clear labeling when using fallback efficiency metric
    - Better documentation of what the proxy metric represents
    """
    print("\n=== Question 10: Efficiency Outliers ===")
    
    # Use PER if available, otherwise create clearly labeled efficiency metric
    if 'season_PER' in df.columns and df['season_PER'].notna().any():
        efficiency = df['season_PER']
        metric_name = 'season_PER'
        print("Using official PER metric for outlier detection")
    else:
        # Create proxy efficiency with clear labeling
        print("PER not available - using simple efficiency proxy")
        print("Simple efficiency = (PTS + AST + TRB) / (FGA + TOV)")
        efficiency = (df['PTS'] + df['AST'] + df['TRB']) / (df['FGA'] + df['TOV']).replace(0, np.nan)
        df['simple_efficiency'] = efficiency
        metric_name = 'simple_efficiency'
    
    # IQR outlier detection
    Q1, Q3 = efficiency.quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    
    # Find outliers
    outlier_mask = (efficiency < lower_bound) | (efficiency > upper_bound)
    outliers = df[outlier_mask].copy()
    outliers['outlier_type'] = np.where(efficiency[outlier_mask] < lower_bound, 'Low', 'High')
    
    result = outliers[['name', 'season', metric_name, 'outlier_type']].sort_values(metric_name, ascending=False)
    
    print(f"Found {len(result)} efficiency outliers using {metric_name}")
    for _, row in result.head(10).iterrows():
        print(f"{row['name']}: {row[metric_name]:.2f} ({row['outlier_type']} outlier)")
    
    return result

def question_11_most_improved(df: pd.DataFrame) -> pd.DataFrame:
    """Question 11: Who improved the most from last season to this season?"""
    print("\n=== Question 11: Most Improved Players ===")
    
    # Step 1: Calculate year-over-year improvement
    df['season_year'] = df['season'].str[:4].astype(int)
    
    # Step 2: Get players with multiple seasons
    player_seasons = df.groupby('player_id')['season_year'].nunique()
    multi_season_players = player_seasons[player_seasons > 1].index
    
    # Step 3: Calculate improvement for multi-season players
    improvements = []
    
    for player_id in multi_season_players:
        player_data = df[df['player_id'] == player_id].sort_values('season_year')
        
        if len(player_data) >= 2:
            current = player_data.iloc[-1]  # Most recent season
            previous = player_data.iloc[-2]  # Previous season
            
            # Calculate improvement in PPG
            ppg_improvement = current['PPG'] - previous['PPG']
            
            improvements.append({
                'player_id': player_id,
                'name': current['name'],
                'previous_season': previous['season'],
                'current_season': current['season'],
                'previous_ppg': previous['PPG'],
                'current_ppg': current['PPG'],
                'ppg_improvement': ppg_improvement
            })
    
    # Step 4: Find most improved
    if improvements:
        result = pd.DataFrame(improvements).nlargest(10, 'ppg_improvement')
        
        print("Most Improved Players (PPG increase):")
        for i, row in result.iterrows():
            print(f"{row['name']}: {row['previous_ppg']:.1f} → {row['current_ppg']:.1f} PPG (+{row['ppg_improvement']:.1f})")
    else:
        result = pd.DataFrame()
        print("No multi-season players found for improvement analysis")
    
    return result

def question_12_pie_top_10(df: pd.DataFrame) -> pd.DataFrame:
    """Use computed season PIE; do not approximate with ad-hoc denominator."""
    print("\n=== Question 12: PIE Top 10 ===")
    pie_col = None
    if "season_PIE" in df.columns:
        pie_col = "season_PIE"
    elif "season_pie" in df.columns:
        pie_col = "season_pie"
    elif "season_pie_pct" in df.columns:
        pie_col = "season_pie_pct"
    else:
        raise ValueError("PIE not found on season table. Ensure compute_pie() ran before this step.")

    result = df.nlargest(10, pie_col)[["name","season",pie_col,"PTS","AST","TRB"]]
    for _, row in result.iterrows():
        v = row[pie_col] * (100 if "pct" in pie_col else 1.0)
        label = "%" if "pct" in pie_col else ""
        print(f"{row['name']}: {v:.3f}{label} PIE")
    return result


# ============================================================================
# ADVANCED QUESTIONS (13-22)
# ============================================================================
def question_13_top_scorers_per_team(df: pd.DataFrame, min_mpg: float = 15.0, k: int = 3) -> pd.DataFrame:
    """
    FIXED VERSION: Uses per-game minutes filter as specified (≥15 MPG, not ≥900 total minutes)
    
    Fixes:
    - Changed from season minutes (≥900) to per-game minutes (≥15 MPG)  
    - Groups by team AND season to avoid conflating multi-season data
    """
    print(f"\n=== Question 13: Top {k} Scorers Per Team (≥{min_mpg} MPG) ===")
    
    df = add_team_column(df.copy())
    
    # Calculate MPG safely
    df['MPG'] = safe_mpg_calculation(df, 'minutes', 'games')
    
    # Add per-36 stats if not present
    if 'PTS_per36' not in df.columns:
        df = add_per36(df, stat_cols=['PTS'], minutes_col='minutes')
    
    # Filter by MPG (per-game criteria, not season total)
    qualified = df[df['MPG'] >= min_mpg].copy()
    print(f"Qualified players: {len(qualified)} (≥{min_mpg} MPG)")
    
    # Group by team AND season (avoid conflating different seasons)
    result = top_k_by_group(qualified, 'PTS_per36', k, group_col=['team', 'season'])
    
    print(f"Top {k} scorers per team-season by points per-36:")
    current_team_season = None
    for _, row in result.iterrows():
        team_season = f"{row['team']} ({row['season']})"
        if team_season != current_team_season:
            current_team_season = team_season
            print(f"\n{team_season}:")
        print(f"  {row['name']}: {row['PTS_per36']:.1f} pts/36 ({row['MPG']:.1f} MPG)")
    
    return result


def question_14_composite_impact_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Composite impact = z(PTS/36) + 0.7*z(AST/36) + 0.7*z(TRB/36)
    - Works on the season-level table using totals + minutes.
    - No reliance on game_id; no re-computation of game-level metrics.
    """
    print("\n=== Question 14: Composite Impact Score ===")
    required = ['minutes', 'PTS', 'AST', 'TRB']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Q14 requires {missing} on the season table.")

    # Ensure per-36 columns exist
    need_per36 = [c for c in ['PTS_per36', 'AST_per36', 'TRB_per36'] if c not in df.columns]
    if need_per36:
        df = add_per36(df, stats=['PTS', 'AST', 'TRB'])

    # Standardize per-36 columns
    df = standardize_columns(df, ['PTS_per36', 'AST_per36', 'TRB_per36'])

    # Compute composite impact
    if not set(['z_PTS_per36', 'z_AST_per36', 'z_TRB_per36']).issubset(df.columns):
        raise RuntimeError("Z-scored per-36 columns not found after standardization.")

    df['impact_score'] = (
        df['z_PTS_per36'] +
        0.7 * df['z_AST_per36'] +
        0.7 * df['z_TRB_per36']
    )

    # Rank (drop NaNs to avoid implicit imputation)
    result = (df[['name', 'season', 'impact_score', 'PTS_per36', 'AST_per36', 'TRB_per36']]
              .dropna(subset=['impact_score'])
              .nlargest(10, 'impact_score'))

    print("Top 10 Composite Impact Scores:")
    for _, row in result.iterrows():
        print(f"{row['name']}: {row['impact_score']:.2f} "
              f"({row['PTS_per36']:.1f}P, {row['AST_per36']:.1f}A, {row['TRB_per36']:.1f}R per-36)")
    return result


def question_15_expected_points_model(df: pd.DataFrame) -> pd.DataFrame:
    """Question 15: Linear model for expected points, find positive residuals"""
    print("\n=== Question 15: Expected Points Model ===")
    
    # Step 1: Prepare data for modeling - drop missing rows instead of zero-fill
    features = ['minutes', 'AST', 'TRB']
    needed = features + ['PTS']
    mask = df[needed].notna().all(axis=1)
    X = df.loc[mask, features]
    y = df.loc[mask, 'PTS']
    
    # Step 2: Fit linear model
    model = LinearRegression()
    model.fit(X, y)
    
    # Step 3: Calculate predictions and residuals
    df = df.copy()
    df.loc[mask, 'predicted_PTS'] = model.predict(X)
    df['PTS_residual'] = df['PTS'] - df['predicted_PTS']
    
    # Step 4: Get top 10 positive residuals (outperformers)
    result = df.loc[mask].nlargest(10, 'PTS_residual')[['name', 'season', 'PTS', 'predicted_PTS', 'PTS_residual']]
    
    print("Top 10 Point Over-Performers (positive residuals):")
    print(f"Model R² = {model.score(X, y):.3f}")
    for i, row in result.iterrows():
        print(f"{row['name']}: {row['PTS']:.0f} actual vs {row['predicted_PTS']:.0f} predicted (+{row['PTS_residual']:.0f})")
    
    return result

def question_16_team_scoring_balance(df: pd.DataFrame) -> pd.DataFrame:
    """
    UPDATED VERSION: Uses unified calculate_team_scoring_balance function
    """
    print("\n=== Question 16: Team Scoring Balance (Detailed) ===")
    
    result = calculate_team_scoring_balance(df, group_cols=['team', 'season'])
    
    print("Team Scoring Distribution (CV = std/mean):")
    print("Lower CV = More Balanced, Higher CV = More Star-Heavy")
    for i, row in result.head(10).iterrows():
        print(f"{row['team']}: CV = {row['cv']:.3f} ({row['balance_type']}, avg {row['mean']:.1f} pts)")
    
    return result

def question_17_minutes_tier_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    UPDATED VERSION: Uses unified helper functions
    
    - Uses safe_mpg_calculation for MPG
    - Uses assign_minutes_tier for tier assignment  
    """
    print("\n=== Question 17: Detailed Minutes Tier Analysis ===")
    
    # Use unified helpers
    df = df.copy()
    df['MPG'] = safe_mpg_calculation(df, 'minutes', 'games')
    df['minutes_tier'] = df['MPG'].apply(assign_minutes_tier)
    
    # Step 2: Calculate comprehensive tier statistics
    tier_stats = (df[df['minutes_tier'] != 'Other']
                  .groupby('minutes_tier')
                  .agg({
                      'PTS': ['mean', 'median', 'std'],
                      'AST': ['mean', 'median', 'std'],
                      'TRB': ['mean', 'median', 'std'],
                      'player_id': 'count'
                  }).round(2))
    
    # Flatten column names
    tier_stats.columns = ['_'.join(col).strip() for col in tier_stats.columns]
    tier_stats = tier_stats.reset_index()
    
    print("Comprehensive minutes tier analysis:")
    for i, row in tier_stats.iterrows():
        print(f"\n{row['minutes_tier']} (n={row['player_id_count']}):")
        print(f"  Points: μ={row['PTS_mean']:.1f}, med={row['PTS_median']:.1f}, σ={row['PTS_std']:.1f}")
        print(f"  Assists: μ={row['AST_mean']:.1f}, med={row['AST_median']:.1f}, σ={row['AST_std']:.1f}")
        print(f"  Rebounds: μ={row['TRB_mean']:.1f}, med={row['TRB_median']:.1f}, σ={row['TRB_std']:.1f}")
    
    return tier_stats

def question_18_three_above_median(df: pd.DataFrame) -> pd.DataFrame:
    """Question 18: Players above team median in points, assists, and rebounds"""
    print("\n=== Question 18: Three-Above-Median Players ===")
    
    # Step 1: Add team info
    df = add_team_column(df.copy())  # Make self-sufficient
    
    # Step 2: Calculate team medians
    team_medians = (df.groupby(['team', 'season'])
                   .agg({'PTS': 'median', 'AST': 'median', 'TRB': 'median'})
                   .add_suffix('_team_median')
                   .reset_index())
    
    # Step 3: Merge with player data
    df_with_medians = df.merge(team_medians, on=['team', 'season'])
    
    # Step 4: Find players above median in all three categories
    above_all_three = df_with_medians[
        (df_with_medians['PTS'] > df_with_medians['PTS_team_median']) &
        (df_with_medians['AST'] > df_with_medians['AST_team_median']) &
        (df_with_medians['TRB'] > df_with_medians['TRB_team_median'])
    ]
    
    result = above_all_three[['name', 'team', 'season', 'PTS', 'AST', 'TRB']].sort_values(['team', 'PTS'], ascending=[True, False])
    
    print("Players above team median in Points, Assists, AND Rebounds:")
    current_team = None
    for i, row in result.iterrows():
        if row['team'] != current_team:
            current_team = row['team']
            print(f"\n{current_team}:")
        print(f"  {row['name']}: {row['PTS']:.0f}P, {row['AST']:.0f}A, {row['TRB']:.0f}R")
    
    return result

def question_19_top_rebounders_per_team(df: pd.DataFrame, k: int = 2) -> pd.DataFrame:
    """Top rebounders per team using reusable function"""
    df = add_team_column(df.copy())  # Make self-sufficient
    return top_k_by_group(df, 'TRB', k, 'team')

def question_20_team_effect_on_scoring(df: pd.DataFrame) -> pd.DataFrame:
    """
    FIXED VERSION: Addresses multicollinearity in Ridge regression
    
    Fixes:
    - Uses fit_intercept=False to avoid perfect multicollinearity
    - Standardizes non-dummy features for better interpretability
    """
    print("\n=== Question 20: Team Effect on Scoring (Fixed Ridge) ===")
    
    df = add_team_column(df.copy())
    
    # Prepare features - drop missing rows
    base_features = ['minutes', 'AST', 'TRB', 'PTS']
    mask = df[base_features].notna().all(axis=1)
    d = df.loc[mask].copy()
    
    # Standardize continuous features for better Ridge interpretability
    continuous_features = ['minutes', 'AST', 'TRB']
    d[continuous_features] = StandardScaler().fit_transform(d[continuous_features])
    
    # Create team dummies
    team_dummies = pd.get_dummies(d['team'], prefix='team')
    X = pd.concat([d[continuous_features], team_dummies], axis=1)
    y = d['PTS']
    
    # FIXED: Use fit_intercept=False to avoid multicollinearity with full dummy encoding
    ridge = Ridge(alpha=1.0, fit_intercept=False)
    ridge.fit(X, y)
    
    # Step 4: Extract team coefficients using pandas operations
    team_coeffs = (
        pd.Series(ridge.coef_, index=X.columns)
        .filter(like='team_')
        .rename_axis('feature')
        .reset_index()
        .assign(team=lambda x: x['feature'].str.replace('team_', '', regex=False))
        .rename(columns={0: 'coefficient'})[['team', 'coefficient']]
        .sort_values('coefficient', ascending=False)
    )
    
    print("Team effects on scoring (Ridge regression, no intercept):")
    print("Positive = team increases scoring, Negative = team decreases scoring")
    print("Note: Coefficients relative to zero (no reference team)")
    for i, row in team_coeffs.iterrows():
        print(f"{row['team']}: {row['coefficient']:+.2f}")
    
    return team_coeffs

def question_21_bootstrap_confidence_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """Question 21: Bootstrap 95% CI for mean points per team"""
    print("\n=== Question 21: Bootstrap 95% CI for Team Scoring ===")
    
    # Step 1: Add team info
    df = add_team_column(df.copy())  # Make self-sufficient
    
    # Step 2: Bootstrap function
    def bootstrap_mean(data, n_bootstrap=1000):
        np.random.seed(42)
        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_means.append(np.mean(sample))
        return np.array(bootstrap_means)
    
    # Step 3: Calculate bootstrap CIs for each team
    team_cis = []
    for team in df['team'].unique():
        team_data = df[df['team'] == team]['PTS'].values
        if len(team_data) > 5:  # Need enough data
            bootstrap_means = bootstrap_mean(team_data)
            ci_lower = np.percentile(bootstrap_means, 2.5)
            ci_upper = np.percentile(bootstrap_means, 97.5)
            observed_mean = np.mean(team_data)
            
            team_cis.append({
                'team': team,
                'mean': observed_mean,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'ci_width': ci_upper - ci_lower,
                'n_players': len(team_data)
            })
    
    result = pd.DataFrame(team_cis).sort_values('mean', ascending=False)
    
    print("Bootstrap 95% Confidence Intervals for Team Mean Scoring:")
    for i, row in result.iterrows():
        print(f"{row['team']}: {row['mean']:.1f} [{row['ci_lower']:.1f}, {row['ci_upper']:.1f}] (n={row['n_players']})")
    
    return result

def question_22_duplicate_detection(df: pd.DataFrame) -> pd.DataFrame:
    """Question 22: Detect and handle duplicate (player_id, name, season) rows"""
    print("\n=== Question 22: Duplicate Detection and Handling ===")
    
    # Step 1: Check for duplicates including season to avoid collapsing multi-season players
    subset = ['player_id', 'name', 'season']
    duplicate_mask = df.duplicated(subset=subset, keep=False)
    duplicates = df[duplicate_mask].copy()
    
    print(f"Found {duplicate_mask.sum()} rows with duplicate (player_id, name, season) combinations")
    
    if len(duplicates) > 0:
        # Step 2: Show duplicate groups
        duplicate_groups = duplicates.groupby(subset).size().reset_index(name='count')
        print(f"Number of duplicate groups: {len(duplicate_groups)}")
        
        # Step 3: Keep the row with max minutes for each duplicate group
        cleaned_df = df.loc[df.groupby(subset)['minutes'].idxmax()]
        
        print(f"After deduplication: {len(df)} → {len(cleaned_df)} rows")
        print("Kept records with maximum minutes played for each player")
        
        # Step 4: Show some examples
        print("\nExample duplicate resolutions:")
        for i, row in duplicate_groups.head(5).iterrows():
            player_duplicates = duplicates[
                (duplicates['player_id'] == row['player_id']) & 
                (duplicates['name'] == row['name']) &
                (duplicates['season'] == row['season'])
            ].sort_values('minutes', ascending=False)
            
            print(f"{row['name']} ({row['player_id']}): {row['count']} duplicates")
            print(f"  Kept: {player_duplicates.iloc[0]['minutes']:.0f} minutes")
            if len(player_duplicates) > 1:
                print(f"  Removed: {player_duplicates.iloc[1]['minutes']:.0f} minutes")
        
        return cleaned_df
    else:
        print("No duplicates found!")
        return df

# ============================================================================
# MAIN EXECUTION WITH ALL QUESTIONS
# ============================================================================

def run_all_interview_questions(df: pd.DataFrame):
    """Run all 22 interview questions in sequence"""
    print("🏀 RUNNING ALL INTERVIEW QUESTIONS")
    print("=" * 60)
    
    results = {}
    
    # Basic Questions (1-12)
    results['q1'] = question_1_top_scorers(df)
    results['q2'] = question_2_double_double_threshold(df)
    results['q3'] = question_3_correlation(df)
    results['q4'] = question_4_true_shooting_leaders(df)
    results['q5'] = question_5_complete_players(df)
    results['q6'] = question_6_team_balance(df)
    df = question_7_clean_fg_percentage(df)  # Modifies df
    results['q8'] = question_8_minutes_tiers(df)
    # results['q9'] = question_9_guards_vs_forwards_scoring(df) # no position column
    results['q10'] = question_10_efficiency_outliers(df)
    results['q11'] = question_11_most_improved(df)
    results['q12'] = question_12_pie_top_10(df)
    
    # Advanced Questions (13-22)
    results['q13'] = question_13_top_scorers_per_team(df)
    results['q14'] = question_14_composite_impact_score(df)
    results['q15'] = question_15_expected_points_model(df)
    results['q16'] = question_16_team_scoring_balance(df)
    results['q17'] = question_17_minutes_tier_analysis(df)
    results['q18'] = question_18_three_above_median(df)
    results['q19'] = question_19_top_rebounders_per_team(df)
    results['q20'] = question_20_team_effect_on_scoring(df)
    results['q21'] = question_21_bootstrap_confidence_intervals(df)
    df_cleaned = question_22_duplicate_detection(df)
    
    print("\n" + "=" * 60)
    print("✅ ALL INTERVIEW QUESTIONS COMPLETED!")
    print("=" * 60)
    
    return results, df_cleaned

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main_analysis():
    """
    ENHANCED VERSION: Main execution with comprehensive debugging
    """
    print("🏀 NBA ANALYSIS PIPELINE - DEBUGGING VERSION")
    print("=" * 70)
    
    base = Path("notebooks/5080_gpu/interview_prep/data/heat_base_data")
    player_df_csv_path = Path(base / "player_statistics_used_Regular Season_from_2009.csv")
    team_df_csv_path = Path(base / "team_statistics_used_Regular Season_from_2009.csv")
    schema_path = Path("notebooks/5080_gpu/interview_prep/data/schema.yaml")  # FIXED: Convert to Path object
    
    # Enhanced file validation with type checking
    print("Validating file paths...")
    files_to_check = [
        (player_df_csv_path, "player data"),
        (team_df_csv_path, "team data"), 
        (schema_path, "schema")
    ]
    
    # DEBUG: Verify all paths are Path objects
    for i, (path, name) in enumerate(files_to_check):
        if not isinstance(path, Path):
            raise TypeError(f"Path {i+1} ({name}) is not a Path object: {type(path)}")
    
    for path, name in files_to_check:
        if not path.exists():
            raise FileNotFoundError(f"{name} file not found: {path}")
        file_size = path.stat().st_size / (1024**2)  # MB
        print(f"✓ Found {name}: {path} ({file_size:.1f} MB)")
    
    # Load data with enhanced error handling
    print(f"\nLoading data...")
    try:
        # Load schema
        schema = load_schema(str(schema_path))
        print(f"✓ Schema loaded successfully")
        print(f"  Numerical columns: {len(schema.numerical)}")
        print(f"  Nominal columns: {len(schema.nominal)}")
        print(f"  ID columns: {len(schema.id_cols)}")
        
        # Load datasets
        player_df = load_data(str(player_df_csv_path), schema)
        team_df = load_data(str(team_df_csv_path), schema)
        
        print(f"✓ Data loading completed:")
        print(f"  Player records: {len(player_df):,}")
        print(f"  Team records: {len(team_df):,}")
        
        # Enhanced data quality assessment
        print(f"\nData quality assessment:")
        
        # Check key columns
        key_player_cols = ['personId', 'gameId', 'numMinutes', 'points', 'season']
        available_key_cols = [col for col in key_player_cols if col in player_df.columns]
        missing_key_cols = [col for col in key_player_cols if col not in player_df.columns]
        
        print(f"  Key columns available: {len(available_key_cols)}/{len(key_player_cols)}")
        if missing_key_cols:
            print(f"  Missing key columns: {missing_key_cols}")
        
        # Check for null values in critical columns
        for col in available_key_cols:
            null_count = player_df[col].isnull().sum()
            null_pct = (null_count / len(player_df)) * 100
            if null_pct > 0:
                print(f"  {col}: {null_count:,} nulls ({null_pct:.1f}%)")
                
    except Exception as e:
        print(f"✗ Data loading error: {e}")
        print(f"Error type: {type(e).__name__}")
        raise
    
    # Calculate metrics with the ENHANCED function
    print(f"\nCalculating metrics...")
    try:
        final_data = calculate_metrics(player_df, team_df)
        print("✓ Metrics calculation completed successfully")
        
        # Enhanced result summary
        print(f"\nFinal dataset summary:")
        print(f"  Total records: {len(final_data):,}")
        print(f"  Unique players: {final_data['player_id'].nunique():,}")
        print(f"  Seasons covered: {sorted(final_data['season'].unique())}")
        print(f"  Total minutes: {final_data['minutes'].sum():,.0f}")
        print(f"  Average minutes per player: {final_data['minutes'].mean():.1f}")
        
        # Check for data completeness
        key_metrics = ['PTS', 'AST', 'TRB', 'minutes']
        for metric in key_metrics:
            if metric in final_data.columns:
                valid_count = final_data[metric].notna().sum()
                valid_pct = (valid_count / len(final_data)) * 100
                print(f"  {metric} completeness: {valid_count:,}/{len(final_data):,} ({valid_pct:.1f}%)")
        
    except Exception as e:
        print(f"✗ Metrics calculation error: {e}")
        print(f"Error type: {type(e).__name__}")
        
        # Enhanced debugging output
        print(f"\nEnhanced debugging information:")
        print(f"Player DF info:")
        print(f"  Shape: {player_df.shape}")
        print(f"  Memory usage: {player_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        print(f"  Columns: {len(player_df.columns)}")
        
        # Show sample of actual data
        print(f"\nSample player data:")
        sample_cols = ['personId', 'gameId', 'numMinutes', 'points'] 
        available_sample_cols = [col for col in sample_cols if col in player_df.columns]
        if available_sample_cols:
            print(player_df[available_sample_cols].head(3).to_string())
        
        raise
    
    # Filter qualified players with enhanced reporting
    print(f"\nFiltering qualified players...")
    min_minutes = 500
    qualified = final_data[final_data['minutes'] >= min_minutes]
    
    filtered_out = len(final_data) - len(qualified)
    filter_pct = (filtered_out / len(final_data)) * 100
    
    print(f"✓ Filtering completed:")
    print(f"  Minimum minutes threshold: {min_minutes}")
    print(f"  Qualified players: {len(qualified):,}")
    print(f"  Filtered out: {filtered_out:,} ({filter_pct:.1f}%)")
    print(f"  Final dataset shape: {qualified.shape}")
    
    return qualified

if __name__ == "__main__":
    data = main_analysis()
    run_all_interview_questions(data)
