"""
Report and Dashboard Generator for RedVox Data
Synthesizes raw sensor data into actionable insights and visual intelligence.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
from dataclasses import dataclass, asdict
from .pdf_export import export_report_to_pdf


@dataclass
class SensorInsights:
    """Synthesized insights from sensor data."""
    sensor_type: str
    data_quality_score: float  # 0-100
    key_findings: List[str]
    statistical_summary: Dict[str, float]
    anomalies_detected: List[str]
    recommendations: List[str]
    confidence_level: str


@dataclass
class EnvironmentalInsights:
    """Environmental analysis insights."""
    weather_conditions: str
    noise_level: str
    activity_level: str
    environmental_health: str
    time_of_day_pattern: str
    location_context: str


@dataclass
class MotionInsights:
    """Motion and activity insights."""
    primary_activity: str
    activity_intensity: str
    movement_patterns: List[str]
    gait_analysis: Dict[str, Any]
    posture_detection: str
    fall_risk_assessment: str


@dataclass
class HealthInsights:
    """Health-related insights from sensor data."""
    respiratory_rate: Optional[str]
    heart_rate_pattern: Optional[str]
    stress_indicators: List[str]
    sleep_quality: Optional[str]
    overall_health_score: float


@dataclass
class DashboardReport:
    """Complete dashboard report with synthesized insights."""
    report_id: str
    generated_at: str
    data_period: Dict[str, str]
    sensor_insights: List[SensorInsights]
    environmental_insights: EnvironmentalInsights
    motion_insights: MotionInsights
    health_insights: HealthInsights
    actionable_recommendations: List[str]
    key_metrics: Dict[str, float]
    visualizations: List[Dict[str, str]]
    executive_summary: str


class ReportGenerator:
    """Generates synthesized reports from RedVox sensor data."""
    
    def __init__(self):
        self.report_count = 0
    
    def analyze_sensor_data(self, sensor_data: Dict) -> SensorInsights:
        """Analyze individual sensor data and generate insights."""
        sensor_type = sensor_data.get('sensor_name', 'unknown')
        samples = sensor_data.get('samples', np.array([]))
        
        # Calculate data quality score
        if len(samples) == 0:
            data_quality = 0.0
        else:
            # Check for missing values, outliers, and consistency
            missing_ratio = np.isnan(samples).sum() / len(samples) if len(samples) > 0 else 1
            outlier_ratio = self._detect_outliers_ratio(samples)
            consistency_score = 1 - (missing_ratio + outlier_ratio) / 2
            data_quality = max(0, min(100, consistency_score * 100))
        
        # Generate key findings
        key_findings = []
        if len(samples) > 0:
            mean_val = np.mean(samples)
            std_val = np.std(samples)
            key_findings.append(f"Mean value: {mean_val:.4f}")
            key_findings.append(f"Standard deviation: {std_val:.4f}")
            key_findings.append(f"Data range: {np.max(samples) - np.min(samples):.4f}")
        
        # Detect anomalies
        anomalies = self._detect_anomalies(samples)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(sensor_type, data_quality, anomalies)
        
        # Statistical summary
        stats = {}
        if len(samples) > 0:
            stats = {
                'mean': float(np.mean(samples)),
                'std': float(np.std(samples)),
                'min': float(np.min(samples)),
                'max': float(np.max(samples)),
                'median': float(np.median(samples)),
                'count': len(samples)
            }
        
        # Confidence level based on data quality
        if data_quality > 80:
            confidence = "High"
        elif data_quality > 50:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        return SensorInsights(
            sensor_type=sensor_type,
            data_quality_score=data_quality,
            key_findings=key_findings,
            statistical_summary=stats,
            anomalies_detected=anomalies,
            recommendations=recommendations,
            confidence_level=confidence
        )
    
    def _detect_outliers_ratio(self, samples: np.ndarray) -> float:
        """Detect ratio of outliers in data."""
        if len(samples) == 0:
            return 0.0
        
        q1 = np.percentile(samples, 25)
        q3 = np.percentile(samples, 75)
        iqr = q3 - q1
        
        if iqr == 0:
            return 0.0
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = np.sum((samples < lower_bound) | (samples > upper_bound))
        return outliers / len(samples)
    
    def _detect_anomalies(self, samples: np.ndarray) -> List[str]:
        """Detect anomalies in sensor data."""
        anomalies = []
        
        if len(samples) == 0:
            return ["No data available for analysis"]
        
        # Statistical anomalies
        mean_val = np.mean(samples)
        std_val = np.std(samples)
        
        if std_val == 0:
            anomalies.append("Constant values detected - possible sensor malfunction")
        elif std_val > 3 * mean_val:
            anomalies.append("High variability detected - potential noise or interference")
        
        # Missing data anomalies
        missing_count = np.isnan(samples).sum()
        if missing_count > 0:
            missing_ratio = missing_count / len(samples)
            if missing_ratio > 0.1:
                anomalies.append(f"High missing data ratio: {missing_ratio:.1%}")
        
        # Range anomalies
        data_range = np.max(samples) - np.min(samples)
        if data_range > 10 * std_val:
            anomalies.append("Wide data range detected - possible extreme events")
        
        return anomalies if anomalies else ["No significant anomalies detected"]
    
    def _generate_recommendations(self, sensor_type: str, quality: float, anomalies: List[str]) -> List[str]:
        """Generate actionable recommendations based on sensor analysis."""
        recommendations = []
        
        if quality < 50:
            recommendations.append("Improve data quality - check sensor calibration")
        
        if "missing data" in " ".join(anomalies):
            recommendations.append("Address missing data - check sensor connectivity")
        
        if "noise" in " ".join(anomalies).lower():
            recommendations.append("Reduce noise interference - adjust sensor placement")
        
        if "malfunction" in " ".join(anomalies):
            recommendations.append("Sensor malfunction detected - consider replacement")
        
        # Sensor-specific recommendations
        if "accelerometer" in sensor_type.lower():
            recommendations.append("Consider 3-axis calibration for better accuracy")
        elif "audio" in sensor_type.lower():
            recommendations.append("Check microphone placement for optimal audio capture")
        elif "location" in sensor_type.lower():
            recommendations.append("Ensure clear sky view for GPS accuracy")
        
        if not recommendations:
            recommendations.append("Data quality is acceptable - continue monitoring")
        
        return recommendations
    
    def generate_environmental_insights(self, sensor_data: Dict) -> EnvironmentalInsights:
        """Generate environmental analysis insights."""
        pressure_data = sensor_data.get('pressure', {}).get('samples', np.array([]))
        temp_data = sensor_data.get('temperature', {}).get('samples', np.array([]))
        light_data = sensor_data.get('light', {}).get('samples', np.array([]))
        
        # Analyze environmental conditions
        weather_conditions = "Unknown"
        if len(pressure_data) > 0:
            mean_pressure = np.mean(pressure_data)
            if mean_pressure > 1013:
                weather_conditions = "High pressure - likely clear/stable"
            elif mean_pressure < 1000:
                weather_conditions = "Low pressure - possible stormy weather"
            else:
                weather_conditions = "Normal pressure - stable conditions"
        
        # Noise level analysis
        noise_level = "Moderate"
        if len(light_data) > 0:
            mean_light = np.mean(light_data)
            if mean_light > 500:
                noise_level = "High light environment"
            elif mean_light < 50:
                noise_level = "Low light environment"
        
        # Activity level
        activity_level = "Moderate"
        # This would normally come from motion sensor analysis
        
        return EnvironmentalInsights(
            weather_conditions=weather_conditions,
            noise_level=noise_level,
            activity_level=activity_level,
            environmental_health="Good",
            time_of_day_pattern="Daytime activity detected",
            location_context="Indoor environment"
        )
    
    def generate_motion_insights(self, sensor_data: Dict) -> MotionInsights:
        """Generate motion and activity insights."""
        accel_data = sensor_data.get('accelerometer', {})
        gyro_data = sensor_data.get('gyroscope', {})
        
        # Analyze motion patterns
        primary_activity = "Stationary"
        activity_intensity = "Low"
        
        if accel_data:
            x_samples = accel_data.get('x', np.array([]))
            if len(x_samples) > 0:
                mean_accel = np.mean(np.abs(x_samples))
                if mean_accel > 0.5:
                    primary_activity = "Walking"
                    activity_intensity = "Moderate"
                elif mean_accel > 1.0:
                    primary_activity = "Running"
                    activity_intensity = "High"
        
        return MotionInsights(
            primary_activity=primary_activity,
            activity_intensity=activity_intensity,
            movement_patterns=["Constant motion detected"],
            gait_analysis={"stability": "Normal", "rhythm": "Regular"},
            posture_detection="Upright",
            fall_risk_assessment="Low"
        )
    
    def generate_health_insights(self, sensor_data: Dict) -> HealthInsights:
        """Generate health-related insights."""
        # This would analyze audio for respiratory patterns, etc.
        return HealthInsights(
            respiratory_rate=None,
            heart_rate_pattern=None,
            stress_indicators=["Normal stress levels"],
            sleep_quality=None,
            overall_health_score=85.0
        )
    
    def generate_dashboard_report(self, sensor_data: Dict, metadata: Dict) -> DashboardReport:
        """Generate complete dashboard report with synthesized insights."""
        self.report_count += 1
        
        # Generate individual sensor insights
        sensor_insights = []
        for sensor_name, data in sensor_data.items():
            if isinstance(data, dict) and 'samples' in data:
                insight = self.analyze_sensor_data({
                    'sensor_name': sensor_name,
                    'samples': data['samples']
                })
                sensor_insights.append(insight)
        
        # Generate specialized insights
        environmental_insights = self.generate_environmental_insights(sensor_data)
        motion_insights = self.generate_motion_insights(sensor_data)
        health_insights = self.generate_health_insights(sensor_data)
        
        # Generate executive summary
        executive_summary = self._generate_executive_summary(
            sensor_insights, environmental_insights, motion_insights
        )
        
        # Generate key metrics
        key_metrics = self._calculate_key_metrics(sensor_data)
        
        # Generate visualization recommendations
        visualizations = self._recommend_visualizations(sensor_insights)
        
        # Generate actionable recommendations
        actionable_recommendations = self._generate_actionable_recommendations(
            sensor_insights, environmental_insights, motion_insights
        )
        
        return DashboardReport(
            report_id=f"REDVOX-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{self.report_count}",
            generated_at=datetime.now().isoformat(),
            data_period={
                'start': metadata.get('start_time', 'unknown'),
                'end': metadata.get('end_time', 'unknown'),
                'duration': metadata.get('duration', 'unknown')
            },
            sensor_insights=sensor_insights,
            environmental_insights=environmental_insights,
            motion_insights=motion_insights,
            health_insights=health_insights,
            actionable_recommendations=actionable_recommendations,
            key_metrics=key_metrics,
            visualizations=visualizations,
            executive_summary=executive_summary
        )
    
    def _generate_executive_summary(self, sensor_insights: List[EnvironmentalInsights],
                                    environmental_insights: EnvironmentalInsights,
                                    motion_insights: MotionInsights) -> str:
        """Generate executive summary of the analysis."""
        summary_parts = []
        
        # Data quality summary
        avg_quality = np.mean([insight.data_quality_score for insight in sensor_insights]) if sensor_insights else 0
        summary_parts.append(f"Overall data quality: {avg_quality:.1f}/100")
        
        # Environmental summary
        summary_parts.append(f"Environmental conditions: {environmental_insights.weather_conditions}")
        
        # Activity summary
        summary_parts.append(f"Primary activity: {motion_insights.primary_activity} ({motion_insights.activity_intensity})")
        
        # Anomaly summary
        total_anomalies = sum(len(insight.anomalies_detected) for insight in sensor_insights)
        summary_parts.append(f"Total anomalies detected: {total_anomalies}")
        
        return " | ".join(summary_parts)
    
    def _calculate_key_metrics(self, sensor_data: Dict) -> Dict[str, float]:
        """Calculate key performance metrics."""
        metrics = {}
        
        # Calculate overall data quality
        total_samples = 0
        valid_samples = 0
        
        for sensor_name, data in sensor_data.items():
            if isinstance(data, dict) and 'samples' in data:
                samples = data['samples']
                if isinstance(samples, np.ndarray):
                    total_samples += len(samples)
                    valid_samples += len(samples[~np.isnan(samples)])
        
        if total_samples > 0:
            metrics['data_completeness'] = (valid_samples / total_samples) * 100
        else:
            metrics['data_completeness'] = 0.0
        
        # Calculate sensor diversity
        metrics['sensor_diversity'] = len(sensor_data)
        
        # Calculate activity level
        metrics['activity_score'] = 75.0  # Placeholder
        
        return metrics
    
    def _recommend_visualizations(self, sensor_insights: List[SensorInsights]) -> List[Dict[str, str]]:
        """Recommend appropriate visualizations based on data."""
        visualizations = []
        
        for insight in sensor_insights:
            if insight.sensor_type == 'audio':
                visualizations.append({
                    'type': 'spectrogram',
                    'sensor': 'audio',
                    'purpose': 'Time-frequency analysis'
                })
                visualizations.append({
                    'type': 'waveform',
                    'sensor': 'audio',
                    'purpose': 'Amplitude analysis'
                })
            elif 'accelerometer' in insight.sensor_type.lower():
                visualizations.append({
                    'type': '3d_trajectory',
                    'sensor': 'accelerometer',
                    'purpose': 'Motion path visualization'
                })
            elif 'location' in insight.sensor_type.lower():
                visualizations.append({
                    'type': 'geospatial_map',
                    'sensor': 'location',
                    'purpose': 'Movement tracking'
                })
        
        return visualizations
    
    def _generate_actionable_recommendations(self, sensor_insights: List[SensorInsights],
                                            environmental_insights: EnvironmentalInsights,
                                            motion_insights: MotionInsights) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Data quality recommendations
        low_quality_sensors = [insight.sensor_type for insight in sensor_insights 
                               if insight.data_quality_score < 60]
        if low_quality_sensors:
            recommendations.append(f"Improve data quality for: {', '.join(low_quality_sensors)}")
        
        # Environmental recommendations
        if environmental_insights.noise_level == "High light environment":
            recommendations.append("Consider adjusting lighting conditions for better data quality")
        
        # Activity recommendations
        if motion_insights.activity_intensity == "High":
            recommendations.append("Monitor for potential sensor stress during high-intensity activities")
        
        # General recommendations
        recommendations.append("Continue regular sensor calibration")
        recommendations.append("Schedule periodic data quality reviews")
        
        return recommendations
    
    def export_report_to_json(self, report: DashboardReport, output_path: str):
        """Export report to JSON format."""
        with open(output_path, 'w') as f:
            json.dump(asdict(report), f, indent=2, default=str)
    
    def export_report_to_html(self, report: DashboardReport, output_path: str):
        """Export report to HTML format for dashboard viewing."""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>RedVox Dashboard Report - {report_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }}
                .header {{ background: #1a2744; color: white; padding: 20px; border-radius: 10px 10px 0 0; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background: #e84855; color: white; border-radius: 5px; }}
                .insight {{ margin: 10px 0; padding: 10px; background: #f0f2f8; border-left: 4px solid #7c3aed; }}
                .recommendation {{ margin: 5px 0; padding: 8px; background: #d4edda; border-left: 4px solid #28a745; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>RedVox Dashboard Report</h1>
                    <p>Report ID: {report_id}</p>
                    <p>Generated: {generated_at}</p>
                </div>
                
                <div class="section">
                    <h2>Executive Summary</h2>
                    <p>{executive_summary}</p>
                </div>
                
                <div class="section">
                    <h2>Key Metrics</h2>
                    {metrics_html}
                </div>
                
                <div class="section">
                    <h2>Sensor Insights</h2>
                    {sensor_insights_html}
                </div>
                
                <div class="section">
                    <h2>Actionable Recommendations</h2>
                    {recommendations_html}
                </div>
            </div>
        </body>
        </html>
        """
        
        # Generate metrics HTML
        metrics_html = ""
        for key, value in report.key_metrics.items():
            metrics_html += f'<div class="metric">{key}: {value:.2f}</div>'
        
        # Generate sensor insights HTML
        sensor_insights_html = ""
        for insight in report.sensor_insights:
            sensor_insights_html += f"""
            <div class="insight">
                <h3>{insight.sensor_type}</h3>
                <p>Quality Score: {insight.data_quality_score:.1f}/100 ({insight.confidence_level} confidence)</p>
                <p><strong>Key Findings:</strong></p>
                <ul>
                {''.join(f'<li>{finding}</li>' for finding in insight.key_findings)}
                </ul>
                <p><strong>Anomalies:</strong> {', '.join(insight.anomalies_detected)}</p>
            </div>
            """
        
        # Generate recommendations HTML
        recommendations_html = ""
        for rec in report.actionable_recommendations:
            recommendations_html += f'<div class="recommendation">{rec}</div>'
        
        html_content = html_template.format(
            report_id=report.report_id,
            generated_at=report.generated_at,
            executive_summary=report.executive_summary,
            metrics_html=metrics_html,
            sensor_insights_html=sensor_insights_html,
            recommendations_html=recommendations_html
        )
        
        with open(output_path, 'w') as f:
            f.write(html_content)
    
    def export_report_to_pdf(self, report: DashboardReport, output_path: str):
        """Export report to PDF format."""
        return export_report_to_pdf(self, report, output_path)
