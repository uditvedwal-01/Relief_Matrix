"""
Machine Learning Models for Disaster Resource Management System
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class DisasterMLPredictor:
    """Main ML predictor class for disaster resource management"""
    
    def __init__(self):
        self.demand_model = None
        self.risk_model = None
        self.trend_model = None
        self.scaler = StandardScaler()
        self.risk_scaler = StandardScaler()
        self.trend_scaler = StandardScaler()
        self.is_trained = False
        self.risk_trained = False
        self.trend_trained = False
        
    def prepare_training_data(self, warehouses, items, beneficiaries, distributions):
        """Prepare training data from disaster data"""
        if not distributions:
            return None, None
            
        # Create features for demand prediction
        features = []
        targets = []
        
        for dist in distributions:
            # Find the item for this distribution
            item = next((i for i in items if i.ItemID == dist.ItemID), None)
            if not item:
                continue
                
            # Find the beneficiary for this distribution
            beneficiary = next((b for b in beneficiaries if b.BeneficiaryID == dist.BeneficiaryID), None)
            if not beneficiary:
                continue
                
            # Create feature vector
            feature_vector = [
                dist.Date.day,  # Day of month
                dist.Date.weekday(),  # Day of week
                item.Quantity,  # Current stock
                len([d for d in distributions if d.ItemID == dist.ItemID]),  # Historical demand count
                len([d for d in distributions if d.BeneficiaryID == dist.BeneficiaryID]),  # Beneficiary activity
                hash(item.Category) % 100 if item.Category else 0,  # Category hash
                len(beneficiaries),  # Total beneficiaries
                len(items),  # Total items
            ]
            
            features.append(feature_vector)
            targets.append(dist.Quantity)
            
        return np.array(features), np.array(targets)
    
    def train_demand_model(self, warehouses, items, beneficiaries, distributions):
        """Train demand forecasting model"""
        X, y = self.prepare_training_data(warehouses, items, beneficiaries, distributions)
        
        if X is None or len(X) < 5:
            # Not enough data for training
            return False
            
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train ensemble model
        self.demand_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        self.demand_model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.demand_model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Demand Model - MAE: {mae:.2f}, R²: {r2:.2f}")
        
        self.is_trained = True
        return True
    
    def train_risk_model(self, disaster, warehouses, items, beneficiaries, distributions):
        """Train risk assessment model using ML"""
        if len(distributions) < 3:
            return False
            
        # Prepare risk features
        risk_features = []
        risk_targets = []
        
        for i in range(len(distributions)):
            # Calculate risk factors
            stock_levels = [item.Quantity for item in items]
            avg_stock = sum(stock_levels) / len(stock_levels) if stock_levels else 0
            low_stock_count = sum(1 for qty in stock_levels if qty < 10)
            
            beneficiary_ratio = len(beneficiaries) / max(1, sum(item.Quantity for item in items))
            recent_activity = len([d for d in distributions if (datetime.now().date() - d.Date).days <= 7])
            
            # Warehouse utilization
            warehouse_util = []
            for warehouse in warehouses:
                warehouse_items = [item for item in items if item.WarehouseID == warehouse.WarehouseID]
                used_capacity = sum(item.Quantity for item in warehouse_items)
                utilization = (used_capacity / warehouse.Capacity * 100) if warehouse.Capacity > 0 else 0
                warehouse_util.append(utilization)
            
            avg_utilization = sum(warehouse_util) / len(warehouse_util) if warehouse_util else 0
            
            # Create feature vector
            feature_vector = [
                avg_stock,
                low_stock_count,
                beneficiary_ratio,
                recent_activity,
                avg_utilization,
                len(distributions),
                len(items),
                len(beneficiaries),
                len(warehouses)
            ]
            
            # Calculate risk score (0-100)
            risk_score = 0
            if low_stock_count > 0:
                risk_score += low_stock_count * 10
            if beneficiary_ratio > 0.1:
                risk_score += min(30, beneficiary_ratio * 100)
            if recent_activity > len(distributions) * 0.8:
                risk_score += 20
            if avg_utilization > 90:
                risk_score += 15
                
            risk_features.append(feature_vector)
            risk_targets.append(min(100, risk_score))
        
        if len(risk_features) < 3:
            return False
            
        X = np.array(risk_features)
        y = np.array(risk_targets)
        
        # Split and scale
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        X_train_scaled = self.risk_scaler.fit_transform(X_train)
        X_test_scaled = self.risk_scaler.transform(X_test)
        
        # Train Random Forest for risk prediction
        self.risk_model = RandomForestRegressor(
            n_estimators=50,
            max_depth=5,
            random_state=42
        )
        
        self.risk_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.risk_model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Risk Model - MAE: {mae:.2f}, R²: {r2:.2f}")
        
        self.risk_trained = True
        return True
    
    def train_trend_model(self, distributions):
        """Train trend analysis model using ML"""
        if len(distributions) < 10:
            return False
            
        # Prepare time series features
        trend_features = []
        trend_targets = []
        
        # Group distributions by date
        daily_distributions = {}
        for dist in distributions:
            date_key = dist.Date
            if date_key not in daily_distributions:
                daily_distributions[date_key] = []
            daily_distributions[date_key].append(dist)
        
        dates = sorted(daily_distributions.keys())
        
        for i in range(7, len(dates)):  # Need at least 7 days of history
            # Features: last 7 days of activity
            recent_days = dates[i-7:i]
            recent_activity = [len(daily_distributions[date]) for date in recent_days]
            
            # Additional features
            day_of_week = dates[i].weekday()
            day_of_month = dates[i].day
            week_number = dates[i].isocalendar()[1]
            
            feature_vector = recent_activity + [day_of_week, day_of_month, week_number]
            target = len(daily_distributions[dates[i]])
            
            trend_features.append(feature_vector)
            trend_targets.append(target)
        
        if len(trend_features) < 5:
            return False
            
        X = np.array(trend_features)
        y = np.array(trend_targets)
        
        # Split and scale
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        X_train_scaled = self.trend_scaler.fit_transform(X_train)
        X_test_scaled = self.trend_scaler.transform(X_test)
        
        # Train Gradient Boosting for trend prediction
        self.trend_model = GradientBoostingRegressor(
            n_estimators=30,
            learning_rate=0.1,
            max_depth=4,
            random_state=42
        )
        
        self.trend_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.trend_model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Trend Model - MAE: {mae:.2f}, R²: {r2:.2f}")
        
        self.trend_trained = True
        return True
    
    def predict_demand(self, items, beneficiaries, distributions, days_ahead=7):
        """Predict resource demand for the next N days"""
        if not self.is_trained or not self.demand_model:
            return self._fallback_predictions(items)
            
        predictions = {}
        current_date = datetime.now().date()
        
        for item in items:
            # Create feature vector for prediction
            historical_demand = len([d for d in distributions if d.ItemID == item.ItemID])
            
            # Predict for each day ahead
            daily_predictions = []
            for day in range(1, days_ahead + 1):
                future_date = current_date + timedelta(days=day)
                
                feature_vector = [
                    future_date.day,
                    future_date.weekday(),
                    item.Quantity,
                    historical_demand,
                    len(beneficiaries),
                    hash(item.Category) % 100 if item.Category else 0,
                    len(beneficiaries),
                    len(items),
                ]
                
                X_pred = self.scaler.transform([feature_vector])
                predicted_demand = max(0, self.demand_model.predict(X_pred)[0])
                daily_predictions.append(predicted_demand)
            
            # Aggregate predictions
            total_predicted = sum(daily_predictions)
            avg_daily = total_predicted / days_ahead
            
            predictions[item.ItemID] = {
                'item_name': item.Name,
                'current_stock': item.Quantity,
                'predicted_demand': int(total_predicted),
                'daily_average': round(avg_daily, 1),
                'stock_status': 'Low' if item.Quantity < total_predicted else 'Adequate',
                'recommendation': self._get_stock_recommendation(item.Quantity, total_predicted)
            }
            
        return predictions
    
    def _fallback_predictions(self, items):
        """Fallback predictions when ML model is not trained"""
        predictions = {}
        
        for item in items:
            # Simple heuristic-based prediction
            predicted_demand = max(item.Quantity * 1.2, 10)  # 20% increase or minimum 10
            
            predictions[item.ItemID] = {
                'item_name': item.Name,
                'current_stock': item.Quantity,
                'predicted_demand': int(predicted_demand),
                'daily_average': round(predicted_demand / 7, 1),
                'stock_status': 'Low' if item.Quantity < predicted_demand else 'Adequate',
                'recommendation': self._get_stock_recommendation(item.Quantity, predicted_demand)
            }
            
        return predictions
    
    def _get_stock_recommendation(self, current_stock, predicted_demand):
        """Get stock management recommendation"""
        if current_stock < predicted_demand * 0.5:
            return "Critical - Immediate restocking required"
        elif current_stock < predicted_demand:
            return "Low - Consider increasing inventory"
        elif current_stock < predicted_demand * 1.5:
            return "Adequate - Monitor closely"
        else:
            return "Good - Well stocked"
    
    def get_allocation_recommendations(self, warehouses, items, beneficiaries, distributions):
        """Get resource allocation recommendations"""
        if not warehouses or not items:
            return []
            
        recommendations = []
        
        # Analyze warehouse utilization
        for warehouse in warehouses:
            warehouse_items = [item for item in items if item.WarehouseID == warehouse.WarehouseID]
            total_capacity = warehouse.Capacity
            used_capacity = sum(item.Quantity for item in warehouse_items)
            utilization = (used_capacity / total_capacity * 100) if total_capacity > 0 else 0
            
            if utilization > 90:
                recommendations.append({
                    'type': 'warning',
                    'message': f"Warehouse {warehouse.Location} is {utilization:.1f}% full - consider redistribution",
                    'priority': 'high'
                })
            elif utilization < 30:
                recommendations.append({
                    'type': 'info',
                    'message': f"Warehouse {warehouse.Location} has {100-utilization:.1f}% free capacity",
                    'priority': 'medium'
                })
        
        # Analyze item distribution
        item_demands = {}
        for dist in distributions:
            item_id = dist.ItemID
            if item_id not in item_demands:
                item_demands[item_id] = 0
            item_demands[item_id] += dist.Quantity
        
        # Find high-demand items
        if item_demands:
            avg_demand = sum(item_demands.values()) / len(item_demands)
            high_demand_items = [item_id for item_id, demand in item_demands.items() if demand > avg_demand * 1.5]
            
            for item_id in high_demand_items:
                item = next((i for i in items if i.ItemID == item_id), None)
                if item:
                    recommendations.append({
                        'type': 'success',
                        'message': f"High demand detected for {item.Name} - ensure adequate supply",
                        'priority': 'high'
                    })
        
        return recommendations
    
    def predict_risk_ml(self, disaster, warehouses, items, beneficiaries, distributions):
        """ML-based risk prediction"""
        if not self.risk_trained or not self.risk_model:
            return self._fallback_risk_assessment(disaster, warehouses, items, beneficiaries, distributions)
        
        # Prepare current features
        stock_levels = [item.Quantity for item in items]
        avg_stock = sum(stock_levels) / len(stock_levels) if stock_levels else 0
        low_stock_count = sum(1 for qty in stock_levels if qty < 10)
        
        beneficiary_ratio = len(beneficiaries) / max(1, sum(item.Quantity for item in items))
        recent_activity = len([d for d in distributions if (datetime.now().date() - d.Date).days <= 7])
        
        warehouse_util = []
        for warehouse in warehouses:
            warehouse_items = [item for item in items if item.WarehouseID == warehouse.WarehouseID]
            used_capacity = sum(item.Quantity for item in warehouse_items)
            utilization = (used_capacity / warehouse.Capacity * 100) if warehouse.Capacity > 0 else 0
            warehouse_util.append(utilization)
        
        avg_utilization = sum(warehouse_util) / len(warehouse_util) if warehouse_util else 0
        
        feature_vector = [
            avg_stock,
            low_stock_count,
            beneficiary_ratio,
            recent_activity,
            avg_utilization,
            len(distributions),
            len(items),
            len(beneficiaries),
            len(warehouses)
        ]
        
        X_pred = self.risk_scaler.transform([feature_vector])
        predicted_risk = max(0, min(100, self.risk_model.predict(X_pred)[0]))
        
        # Determine risk level
        if predicted_risk >= 70:
            risk_level = "High"
        elif predicted_risk >= 40:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        # Generate risk factors based on ML insights
        risk_factors = []
        if low_stock_count > 0:
            risk_factors.append(f"{low_stock_count} items with low stock")
        if beneficiary_ratio > 0.1:
            risk_factors.append("High beneficiary to resource ratio")
        if recent_activity > len(distributions) * 0.8:
            risk_factors.append("High recent distribution activity")
        if avg_utilization > 90:
            risk_factors.append("Warehouse capacity near limit")
        
        return {
            'risk_level': risk_level,
            'risk_score': int(predicted_risk),
            'risk_factors': risk_factors,
            'recommendations': self._get_risk_recommendations(risk_level, risk_factors),
            'ml_trained': True
        }
    
    def predict_trend_ml(self, distributions, days_ahead=7):
        """ML-based trend prediction"""
        if not self.trend_trained or not self.trend_model or len(distributions) < 7:
            return self._fallback_trend_analysis(distributions)
        
        # Group distributions by date
        daily_distributions = {}
        for dist in distributions:
            date_key = dist.Date
            if date_key not in daily_distributions:
                daily_distributions[date_key] = []
            daily_distributions[date_key].append(dist)
        
        dates = sorted(daily_distributions.keys())
        
        # Get last 7 days of activity
        recent_days = dates[-7:] if len(dates) >= 7 else dates
        recent_activity = [len(daily_distributions[date]) for date in recent_days]
        
        # Pad with zeros if needed
        while len(recent_activity) < 7:
            recent_activity.insert(0, 0)
        
        # Predict next 7 days
        predictions = []
        for day in range(days_ahead):
            future_date = datetime.now().date() + timedelta(days=day)
            day_of_week = future_date.weekday()
            day_of_month = future_date.day
            week_number = future_date.isocalendar()[1]
            
            feature_vector = recent_activity + [day_of_week, day_of_month, week_number]
            X_pred = self.trend_scaler.transform([feature_vector])
            predicted_activity = max(0, self.trend_model.predict(X_pred)[0])
            predictions.append(int(predicted_activity))
            
            # Update recent activity for next prediction
            recent_activity = recent_activity[1:] + [predicted_activity]
        
        total_predicted = sum(predictions)
        avg_daily = total_predicted / days_ahead
        
        # Determine trend direction
        if len(dates) >= 2:
            recent_avg = sum(recent_activity[-3:]) / 3 if len(recent_activity) >= 3 else sum(recent_activity) / len(recent_activity)
            if avg_daily > recent_avg * 1.1:
                trend_direction = "increasing"
            elif avg_daily < recent_avg * 0.9:
                trend_direction = "decreasing"
            else:
                trend_direction = "stable"
        else:
            trend_direction = "stable"
        
        return {
            'trend': f'ML Prediction: {total_predicted} distributions in next {days_ahead} days',
            'total_distributions': total_predicted,
            'avg_daily': round(avg_daily, 1),
            'trend_direction': trend_direction,
            'daily_predictions': predictions,
            'ml_trained': True
        }
    
    def get_risk_assessment(self, disaster, warehouses, items, beneficiaries, distributions):
        """Generate risk assessment using ML if available, fallback otherwise"""
        if self.risk_trained and self.risk_model:
            return self.predict_risk_ml(disaster, warehouses, items, beneficiaries, distributions)
        else:
            return self._fallback_risk_assessment(disaster, warehouses, items, beneficiaries, distributions)
    
    def _fallback_risk_assessment(self, disaster, warehouses, items, beneficiaries, distributions):
        """Fallback risk assessment when ML is not available"""
        risk_factors = []
        risk_score = 0
        
        # Factor 1: Stock levels
        low_stock_items = [item for item in items if item.Quantity < 10]
        if low_stock_items:
            risk_factors.append(f"{len(low_stock_items)} items with low stock")
            risk_score += len(low_stock_items) * 10
        
        # Factor 2: Beneficiary to resource ratio
        total_resources = sum(item.Quantity for item in items)
        if beneficiaries and total_resources > 0:
            ratio = len(beneficiaries) / total_resources
            if ratio > 0.1:  # More than 1 beneficiary per 10 resources
                risk_factors.append("High beneficiary to resource ratio")
                risk_score += 20
        
        # Factor 3: Distribution activity
        recent_distributions = [d for d in distributions if (datetime.now().date() - d.Date).days <= 7]
        if len(recent_distributions) > len(distributions) * 0.8:
            risk_factors.append("High recent distribution activity")
            risk_score += 15
        
        # Factor 4: Warehouse capacity
        for warehouse in warehouses:
            warehouse_items = [item for item in items if item.WarehouseID == warehouse.WarehouseID]
            used_capacity = sum(item.Quantity for item in warehouse_items)
            if warehouse.Capacity > 0 and (used_capacity / warehouse.Capacity) > 0.9:
                risk_factors.append(f"Warehouse {warehouse.Location} near capacity")
                risk_score += 10
        
        # Determine risk level
        if risk_score >= 50:
            risk_level = "High"
        elif risk_score >= 25:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        return {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'risk_factors': risk_factors,
            'recommendations': self._get_risk_recommendations(risk_level, risk_factors),
            'ml_trained': False
        }
    
    def _fallback_trend_analysis(self, distributions):
        """Fallback trend analysis when ML is not available"""
        if not distributions:
            return {
                'trend': 'No data',
                'total_distributions': 0,
                'avg_daily': 0,
                'trend_direction': 'stable',
                'ml_trained': False
            }
        
        # Simple trend calculation
        recent_distributions = [d for d in distributions if (datetime.now().date() - d.Date).days <= 30]
        total_distributions = len(recent_distributions)
        avg_daily = total_distributions / 30 if total_distributions > 0 else 0
        
        return {
            'trend': f'{total_distributions} distributions in last 30 days',
            'total_distributions': total_distributions,
            'avg_daily': round(avg_daily, 1),
            'trend_direction': 'stable',
            'ml_trained': False
        }
    
    def _get_risk_recommendations(self, risk_level, risk_factors):
        """Get recommendations based on risk assessment"""
        recommendations = []
        
        if risk_level == "High":
            recommendations.extend([
                "Immediate action required - review all stock levels",
                "Consider emergency procurement",
                "Implement priority distribution system"
            ])
        elif risk_level == "Medium":
            recommendations.extend([
                "Monitor stock levels closely",
                "Prepare contingency plans",
                "Review distribution priorities"
            ])
        else:
            recommendations.extend([
                "Continue regular monitoring",
                "Maintain current stock levels",
                "Plan for future needs"
            ])
        
        return recommendations
    


# Global ML predictor instance
ml_predictor = DisasterMLPredictor()
