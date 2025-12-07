"""
Bundle Recommendation Engine using Association Rules and Clustering
Includes: Apriori, FP-Growth, Eclat, and clustering-based recommendations
"""

import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class BundleRecommender:
    """
    Comprehensive bundle recommendation system using multiple techniques
    """
    
    def __init__(self, min_support=0.01, min_confidence=0.3, min_lift=1.2):
        """
        Initialize bundle recommender
        
        Args:
            min_support: Minimum support threshold for item sets
            min_confidence: Minimum confidence for rules
            min_lift: Minimum lift for rules
        """
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_lift = min_lift
        self.rules = None
        self.clusters = None
        self.recommendations = []
        
    def prepare_basket_data(self, df):
        """
        Transform order data into basket format for association rules
        
        Args:
            df: DataFrame with order_id and pizza_name columns
            
        Returns:
            Basket matrix (one-hot encoded)
        """
        # Create basket by order_id
        basket = df.groupby(['order_id', 'pizza_name'])['quantity'].sum().unstack().fillna(0)
        
        # Convert to binary (0/1) matrix
        basket_sets = basket.applymap(lambda x: 1 if x > 0 else 0)
        
        print(f"📦 Created basket matrix: {basket_sets.shape[0]} transactions × {basket_sets.shape[1]} items")
        
        return basket_sets
    
    def find_association_rules_apriori(self, basket_data):
        """
        Find association rules using Apriori algorithm
        
        Args:
            basket_data: Binary basket matrix
            
        Returns:
            DataFrame of association rules
        """
        print("\n🔍 Running Apriori Algorithm...")
        
        # Find frequent itemsets
        frequent_itemsets = apriori(basket_data, 
                                   min_support=self.min_support, 
                                   use_colnames=True)
        
        if len(frequent_itemsets) == 0:
            print("⚠️ No frequent itemsets found. Try lowering min_support.")
            return pd.DataFrame()
        
        # Generate rules
        rules = association_rules(frequent_itemsets, 
                                 metric="confidence", 
                                 min_threshold=self.min_confidence)
        
        # Filter by lift
        rules = rules[rules['lift'] >= self.min_lift]
        
        # Sort by lift
        rules = rules.sort_values('lift', ascending=False)
        
        print(f"✅ Found {len(rules)} association rules")
        
        self.rules = rules
        return rules
    
    def find_association_rules_fpgrowth(self, basket_data):
        """
        Find association rules using FP-Growth algorithm (faster than Apriori)
        
        Args:
            basket_data: Binary basket matrix
            
        Returns:
            DataFrame of association rules
        """
        print("\n🔍 Running FP-Growth Algorithm...")
        
        # Find frequent itemsets using FP-Growth
        frequent_itemsets = fpgrowth(basket_data, 
                                    min_support=self.min_support, 
                                    use_colnames=True)
        
        if len(frequent_itemsets) == 0:
            print("⚠️ No frequent itemsets found. Try lowering min_support.")
            return pd.DataFrame()
        
        # Generate rules
        rules = association_rules(frequent_itemsets, 
                                 metric="confidence", 
                                 min_threshold=self.min_confidence)
        
        # Filter by lift
        rules = rules[rules['lift'] >= self.min_lift]
        
        # Sort by lift
        rules = rules.sort_values('lift', ascending=False)
        
        print(f"✅ Found {len(rules)} association rules")
        
        return rules
    
    def generate_bundle_recommendations(self, rules_df, top_n=10):
        """
        Generate bundle recommendations with business reasoning
        
        Args:
            rules_df: DataFrame of association rules
            top_n: Number of top recommendations to generate
            
        Returns:
            List of bundle recommendations with explanations
        """
        recommendations = []
        
        if rules_df.empty:
            return recommendations
        
        for idx, row in rules_df.head(top_n).iterrows():
            # Extract antecedents and consequents
            antecedents = list(row['antecedents'])
            consequents = list(row['consequents'])
            
            # Create bundle
            bundle = antecedents + consequents
            
            # Generate recommendation
            rec = {
                'bundle_id': idx + 1,
                'bundle_items': bundle,
                'if_customer_buys': antecedents,
                'recommend': consequents,
                'support': round(row['support'] * 100, 2),
                'confidence': round(row['confidence'] * 100, 2),
                'lift': round(row['lift'], 2),
                'expected_uplift': round((row['lift'] - 1) * 100, 2),
                'reasoning': self._generate_reasoning(row)
            }
            
            recommendations.append(rec)
        
        self.recommendations = recommendations
        return recommendations
    
    def _generate_reasoning(self, rule):
        """
        Generate human-readable reasoning for a rule
        
        Args:
            rule: Association rule row
            
        Returns:
            String explanation
        """
        antecedents = ', '.join(list(rule['antecedents']))
        consequents = ', '.join(list(rule['consequents']))
        
        reasoning = f"When customers order {antecedents}, they are {rule['lift']:.1f}x more likely "
        reasoning += f"to also order {consequents}. "
        reasoning += f"This bundle appears in {rule['support']*100:.1f}% of all orders "
        reasoning += f"with {rule['confidence']*100:.1f}% confidence. "
        
        if rule['lift'] > 2:
            reasoning += "This is a STRONG association - highly recommended for bundling!"
        elif rule['lift'] > 1.5:
            reasoning += "This is a good association for bundle promotions."
        else:
            reasoning += "This shows moderate association worth testing."
        
        return reasoning
    
    def cluster_based_bundles(self, df, n_clusters=5):
        """
        Create bundles based on clustering of pizza characteristics
        
        Args:
            df: DataFrame with pizza features
            n_clusters: Number of clusters to create
            
        Returns:
            Cluster-based bundle recommendations
        """
        print("\n🎨 Creating cluster-based bundles...")
        
        # Prepare features for clustering
        features = ['unit_price', 'ingredient_count']
        if 'size_numeric' in df.columns:
            features.append('size_numeric')
        
        # Aggregate by pizza
        pizza_features = df.groupby('pizza_name')[features].mean()
        
        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(pizza_features)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        pizza_features['cluster'] = kmeans.fit_predict(scaled_features)
        
        # Create bundles from clusters
        cluster_bundles = []
        for cluster_id in range(n_clusters):
            cluster_pizzas = pizza_features[pizza_features['cluster'] == cluster_id].index.tolist()
            
            if len(cluster_pizzas) >= 2:
                # Calculate cluster statistics
                cluster_data = df[df['pizza_name'].isin(cluster_pizzas)]
                avg_price = cluster_data['unit_price'].mean()
                total_orders = len(cluster_data)
                
                bundle = {
                    'cluster_id': cluster_id,
                    'bundle_type': self._classify_cluster(avg_price, cluster_id),
                    'pizzas': cluster_pizzas[:5],  # Top 5 pizzas from cluster
                    'avg_price': round(avg_price, 2),
                    'popularity': total_orders,
                    'recommendation': f"Bundle {len(cluster_pizzas[:3])} items from this group for variety"
                }
                cluster_bundles.append(bundle)
        
        self.clusters = cluster_bundles
        return cluster_bundles
    
    def _classify_cluster(self, avg_price, cluster_id):
        """Classify cluster based on characteristics"""
        if avg_price < 12:
            return "Budget Friendly"
        elif avg_price < 16:
            return "Popular Favorites"
        elif avg_price < 20:
            return "Premium Selection"
        else:
            return "Gourmet Collection"
    
    def get_time_based_bundles(self, df):
        """
        Generate bundles based on time patterns (lunch, dinner, weekend)
        
        Args:
            df: DataFrame with time features
            
        Returns:
            Time-based bundle recommendations
        """
        time_bundles = []
        
        # Lunch bundles
        lunch_data = df[df['meal_period'] == 'Lunch']
        if len(lunch_data) > 0:
            lunch_top = lunch_data['pizza_name'].value_counts().head(3).index.tolist()
            time_bundles.append({
                'period': 'Lunch Special',
                'bundle': lunch_top,
                'reasoning': 'Most popular items during lunch hours',
                'suggested_discount': '15% off when ordered 11am-2pm'
            })
        
        # Dinner bundles
        dinner_data = df[df['meal_period'] == 'Dinner']
        if len(dinner_data) > 0:
            dinner_top = dinner_data['pizza_name'].value_counts().head(3).index.tolist()
            time_bundles.append({
                'period': 'Dinner Deal',
                'bundle': dinner_top,
                'reasoning': 'Top sellers during dinner rush',
                'suggested_discount': 'Family deal: Buy 2 get 20% off'
            })
        
        # Weekend bundles
        weekend_data = df[df['is_weekend'] == 1]
        if len(weekend_data) > 0:
            weekend_top = weekend_data['pizza_name'].value_counts().head(4).index.tolist()
            time_bundles.append({
                'period': 'Weekend Party Pack',
                'bundle': weekend_top,
                'reasoning': 'Popular weekend combinations',
                'suggested_discount': 'Weekend special: 4 pizzas for price of 3'
            })
        
        return time_bundles
    
    def compare_algorithms(self, basket_data):
        """
        Compare Apriori and FP-Growth algorithms
        
        Args:
            basket_data: Binary basket matrix
            
        Returns:
            Comparison results
        """
        import time
        
        results = []
        
        # Test Apriori
        start_time = time.time()
        apriori_rules = self.find_association_rules_apriori(basket_data)
        apriori_time = time.time() - start_time
        
        results.append({
            'Algorithm': 'Apriori',
            'Rules Found': len(apriori_rules) if not apriori_rules.empty else 0,
            'Time (seconds)': round(apriori_time, 3),
            'Avg Lift': round(apriori_rules['lift'].mean(), 2) if not apriori_rules.empty else 0,
            'Max Lift': round(apriori_rules['lift'].max(), 2) if not apriori_rules.empty else 0
        })
        
        # Test FP-Growth
        start_time = time.time()
        fpgrowth_rules = self.find_association_rules_fpgrowth(basket_data)
        fpgrowth_time = time.time() - start_time
        
        results.append({
            'Algorithm': 'FP-Growth',
            'Rules Found': len(fpgrowth_rules) if not fpgrowth_rules.empty else 0,
            'Time (seconds)': round(fpgrowth_time, 3),
            'Avg Lift': round(fpgrowth_rules['lift'].mean(), 2) if not fpgrowth_rules.empty else 0,
            'Max Lift': round(fpgrowth_rules['lift'].max(), 2) if not fpgrowth_rules.empty else 0
        })
        
        comparison_df = pd.DataFrame(results)
        print("\n📊 Algorithm Comparison:")
        print(comparison_df.to_string(index=False))
        
        # Select best based on rules found and time
        if results[0]['Rules Found'] > results[1]['Rules Found']:
            print(f"\n🏆 Apriori found more rules ({results[0]['Rules Found']} vs {results[1]['Rules Found']})")
        elif results[1]['Rules Found'] > results[0]['Rules Found']:
            print(f"\n🏆 FP-Growth found more rules ({results[1]['Rules Found']} vs {results[0]['Rules Found']})")
        else:
            if fpgrowth_time < apriori_time:
                print(f"\n🏆 FP-Growth is faster ({fpgrowth_time:.3f}s vs {apriori_time:.3f}s)")
            else:
                print(f"\n🏆 Both algorithms performed similarly")
        
        return comparison_df
