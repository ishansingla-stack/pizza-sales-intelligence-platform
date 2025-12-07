"""
Feature Engineering Module for Pizza Sales Intelligence
Includes: Time features, Phoenix seasonality, business metrics, ingredient mapping
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import holidays
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """
    Comprehensive feature engineering for pizza sales analysis
    """
    
    def __init__(self):
        # Initialize US holidays with Phoenix-specific considerations
        self.us_holidays = holidays.US(state='AZ', years=range(2010, 2030))
        self.phoenix_seasons = self._define_phoenix_seasons()
        
    def _define_phoenix_seasons(self) -> Dict:
        """Define Phoenix-specific seasonal patterns"""
        return {
            'extreme_heat': {'months': [6, 7, 8, 9], 'impact': 'negative'},
            'perfect_weather': {'months': [3, 4, 10, 11], 'impact': 'positive'},
            'snowbird_season': {'months': [1, 2, 12], 'impact': 'positive'},
            'tourist_peak': {'months': [1, 2, 3], 'impact': 'positive'},
            'monsoon_season': {'months': [7, 8], 'impact': 'mixed'}
        }
    
    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering transformations
        
        Args:
            df: Input dataframe with raw pizza sales data
            
        Returns:
            DataFrame with engineered features
        """
        print("🔧 Starting feature engineering...")
        
        # Create a copy to avoid modifying original
        df_features = df.copy()
        
        # 1. Time-based features
        df_features = self.create_time_features(df_features)
        
        # 2. Phoenix-specific features
        df_features = self.create_phoenix_features(df_features)
        
        # 3. Business features
        df_features = self.create_business_features(df_features)
        
        # 4. Ingredient features
        df_features = self.create_ingredient_features(df_features)
        
        # 5. Cyclical encoding for time
        df_features = self.create_cyclical_features(df_features)
        
        print(f"✅ Feature engineering complete! Created {len(df_features.columns) - len(df.columns)} new features")
        
        return df_features
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive time-based features"""
        
        # Ensure datetime format
        df['order_date'] = pd.to_datetime(df['order_date'])
        
        # Extract time components
        df['year'] = df['order_date'].dt.year
        df['month'] = df['order_date'].dt.month
        df['day'] = df['order_date'].dt.day
        df['dayofweek'] = df['order_date'].dt.dayofweek
        df['day_name'] = df['order_date'].dt.day_name()
        df['week'] = df['order_date'].dt.isocalendar().week
        df['quarter'] = df['order_date'].dt.quarter
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        df['is_month_start'] = df['order_date'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['order_date'].dt.is_month_end.astype(int)
        
        # Hour extraction from order_time
        if df['order_time'].dtype == 'object':
            df['hour'] = pd.to_datetime(df['order_time'], format='%H:%M:%S').dt.hour
        else:
            df['hour'] = df['order_time'].apply(lambda x: x.hour if hasattr(x, 'hour') else 0)
        
        # Meal periods
        df['meal_period'] = pd.cut(df['hour'], 
                                   bins=[0, 11, 14, 17, 21, 24],
                                   labels=['Morning', 'Lunch', 'Afternoon', 'Dinner', 'Late Night'],
                                   include_lowest=True)
        
        # Business hours
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
        
        # Days from payday (assuming 1st and 15th)
        df['days_from_payday'] = df['day'].apply(lambda x: min(abs(x - 1), abs(x - 15), abs(x - 30)))
        df['is_payday_week'] = (df['days_from_payday'] <= 3).astype(int)
        
        return df
    
    def create_phoenix_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create Phoenix, Arizona specific features"""
        
        # Holiday features
        df['is_holiday'] = df['order_date'].apply(lambda x: 1 if x in self.us_holidays else 0)
        df['days_to_holiday'] = df['order_date'].apply(self._days_to_nearest_holiday)
        df['holiday_name'] = df['order_date'].apply(
            lambda x: self.us_holidays.get(x, 'None')
        )
        
        # Phoenix seasonal patterns
        df['phoenix_season'] = df['month'].apply(self._get_phoenix_season)
        df['is_extreme_heat'] = df['month'].isin([6, 7, 8, 9]).astype(int)
        df['is_perfect_weather'] = df['month'].isin([3, 4, 10, 11]).astype(int)
        df['is_snowbird_season'] = df['month'].isin([1, 2, 12]).astype(int)
        df['is_monsoon_season'] = df['month'].isin([7, 8]).astype(int)
        
        # Major Phoenix events (approximate)
        df['is_spring_training'] = ((df['month'] == 3) | 
                                    ((df['month'] == 2) & (df['day'] >= 15))).astype(int)
        df['is_waste_management_open'] = ((df['month'] == 2) & 
                                          (df['week'] == 5)).astype(int)
        
        # Temperature impact (simulated based on month)
        temp_map = {1: 65, 2: 70, 3: 75, 4: 85, 5: 95, 6: 105,
                   7: 106, 8: 105, 9: 100, 10: 88, 11: 75, 12: 65}
        df['avg_temp_estimate'] = df['month'].map(temp_map)
        df['is_too_hot'] = (df['avg_temp_estimate'] >= 100).astype(int)
        
        # School calendar impacts
        df['is_summer_break'] = ((df['month'] >= 6) & (df['month'] <= 7)).astype(int)
        df['is_winter_break'] = ((df['month'] == 12) & (df['day'] >= 20)).astype(int)
        df['is_spring_break'] = ((df['month'] == 3) & (df['week'] == 2)).astype(int)
        
        return df
    
    def create_business_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create business-related features"""
        
        # Price features
        df['price_per_unit'] = df['total_price'] / df['quantity']
        df['is_bulk_order'] = (df['quantity'] >= 3).astype(int)
        
        # Size encoding (ordinal)
        size_map = {'S': 1, 'M': 2, 'L': 3, 'XL': 4, 'XXL': 5}
        df['size_numeric'] = df['pizza_size'].map(size_map)
        
        # Category encoding (one-hot)
        category_dummies = pd.get_dummies(df['pizza_category'], prefix='category')
        df = pd.concat([df, category_dummies], axis=1)
        
        # Price buckets
        df['price_bucket'] = pd.qcut(df['unit_price'], 
                                     q=4, 
                                     labels=['Budget', 'Standard', 'Premium', 'Luxury'])
        
        # Order value category
        df['order_value_category'] = pd.cut(df['total_price'],
                                           bins=[0, 15, 30, 50, 100, 1000],
                                           labels=['Small', 'Medium', 'Large', 'XLarge', 'Bulk'])
        
        # Pizza complexity (ingredient count)
        df['ingredient_count'] = df['pizza_ingredients'].apply(
            lambda x: len(x.split(',')) if pd.notna(x) else 0
        )
        df['is_complex_pizza'] = (df['ingredient_count'] >= 5).astype(int)
        
        return df
    
    def create_ingredient_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract and encode ingredient information"""
        
        # Extract individual ingredients
        all_ingredients = set()
        for ingredients in df['pizza_ingredients'].dropna():
            all_ingredients.update([i.strip() for i in ingredients.split(',')])
        
        # Create binary features for top ingredients
        top_ingredients = ['Mozzarella Cheese', 'Tomatoes', 'Red Onions', 
                          'Red Peppers', 'Mushrooms', 'Pepperoni', 'Chicken',
                          'Bacon', 'Garlic', 'Green Peppers']
        
        for ingredient in top_ingredients:
            col_name = f"has_{ingredient.lower().replace(' ', '_')}"
            df[col_name] = df['pizza_ingredients'].apply(
                lambda x: 1 if pd.notna(x) and ingredient in x else 0
            )
        
        # Dietary categories
        df['is_vegetarian'] = (~df['pizza_ingredients'].str.contains(
            'Chicken|Pepperoni|Bacon|Ham|Beef|Salami|Prosciutto|Anchovies',
            na=False, case=False
        )).astype(int)
        
        df['has_meat'] = df['pizza_ingredients'].str.contains(
            'Chicken|Pepperoni|Bacon|Ham|Beef|Salami|Prosciutto|Anchovies',
            na=False, case=False
        ).astype(int)
        
        df['has_seafood'] = df['pizza_ingredients'].str.contains(
            'Anchovies|Shrimp|Salmon|Tuna',
            na=False, case=False
        ).astype(int)
        
        df['has_cheese'] = df['pizza_ingredients'].str.contains(
            'Cheese|Mozzarella|Provolone|Romano|Feta|Asiago|Ricotta',
            na=False, case=False
        ).astype(int)
        
        return df
    
    def create_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create cyclical encodings for temporal features"""
        
        # Hour of day (24-hour cycle)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Day of week (7-day cycle)
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        
        # Day of month (30-day cycle)
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 30)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 30)
        
        # Month of year (12-month cycle)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Week of year (52-week cycle)
        df['week_sin'] = np.sin(2 * np.pi * df['week'] / 52)
        df['week_cos'] = np.cos(2 * np.pi * df['week'] / 52)
        
        return df
    
    def _days_to_nearest_holiday(self, date):
        """Calculate days to nearest holiday"""
        year = date.year
        holidays_list = [d for d in self.us_holidays if d.year == year]
        if not holidays_list:
            return 30
        
        days_diff = [(holiday - date).days for holiday in holidays_list]
        future_holidays = [d for d in days_diff if d >= 0]
        
        if future_holidays:
            return min(future_holidays)
        else:
            return 30
    
    def _get_phoenix_season(self, month):
        """Get Phoenix season based on month"""
        if month in [6, 7, 8, 9]:
            return 'Extreme Heat'
        elif month in [3, 4, 10, 11]:
            return 'Perfect Weather'
        elif month in [1, 2, 12]:
            return 'Snowbird Season'
        else:
            return 'Transition'
    
    def get_feature_summary(self, df: pd.DataFrame) -> Dict:
        """Get summary of engineered features"""
        
        feature_groups = {
            'time_features': ['year', 'month', 'day', 'hour', 'dayofweek', 'week', 'quarter'],
            'phoenix_features': ['is_extreme_heat', 'is_perfect_weather', 'is_snowbird_season'],
            'business_features': ['price_per_unit', 'is_bulk_order', 'size_numeric'],
            'ingredient_features': ['ingredient_count', 'is_vegetarian', 'has_meat'],
            'cyclical_features': ['hour_sin', 'hour_cos', 'month_sin', 'month_cos']
        }
        
        summary = {}
        for group, features in feature_groups.items():
            available = [f for f in features if f in df.columns]
            summary[group] = {
                'count': len(available),
                'features': available
            }
        
        return summary
