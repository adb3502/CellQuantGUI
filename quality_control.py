# Quality Control and Validation Module
# Ensures data quality and analysis reliability through comprehensive validation

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import logging
import warnings
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import sklearn.metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

@dataclass
class QualityMetric:
    """Represents a quality control metric"""
    name: str
    value: float
    threshold: float
    status: str  # 'pass', 'warning', 'fail'
    description: str
    recommendation: str = ""

@dataclass
class ValidationResult:
    """Result of a validation check"""
    check_name: str
    status: str  # 'pass', 'warning', 'fail'
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class QualityReport:
    """Comprehensive quality control report"""
    experiment_name: str
    timestamp: datetime
    overall_status: str
    metrics: List[QualityMetric]
    validation_results: List[ValidationResult]
    recommendations: List[str]
    data_summary: Dict[str, Any]

class ImageQualityAnalyzer:
    """Analyzes image quality metrics"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_image_quality(self, image: np.ndarray, channel_name: str = "") -> Dict[str, float]:
        """Analyze various image quality metrics"""
        
        metrics = {}
        
        try:
            # Basic intensity statistics
            metrics['mean_intensity'] = float(np.mean(image))
            metrics['std_intensity'] = float(np.std(image))
            metrics['min_intensity'] = float(np.min(image))
            metrics['max_intensity'] = float(np.max(image))
            metrics['dynamic_range'] = metrics['max_intensity'] - metrics['min_intensity']
            
            # Signal-to-noise ratio estimation
            metrics['snr'] = self._estimate_snr(image)
            
            # Contrast metrics
            metrics['rms_contrast'] = self._calculate_rms_contrast(image)
            metrics['michelson_contrast'] = self._calculate_michelson_contrast(image)
            
            # Sharpness/focus metrics
            metrics['gradient_magnitude'] = self._calculate_gradient_magnitude(image)
            metrics['laplacian_variance'] = self._calculate_laplacian_variance(image)
            
            # Noise estimation
            metrics['noise_level'] = self._estimate_noise_level(image)
            
            # Saturation check
            metrics['saturation_percentage'] = self._calculate_saturation_percentage(image)
            
            # Uniformity metrics
            metrics['coefficient_of_variation'] = metrics['std_intensity'] / metrics['mean_intensity'] if metrics['mean_intensity'] > 0 else 0
            
        except Exception as e:
            self.logger.error(f"Error analyzing image quality for {channel_name}: {e}")
        
        return metrics
    
    def _estimate_snr(self, image: np.ndarray) -> float:
        """Estimate signal-to-noise ratio"""
        try:
            # Simple SNR estimation using signal vs background regions
            # Assume top 25% of intensities are signal, bottom 25% are background
            sorted_intensities = np.sort(image.flatten())
            n_pixels = len(sorted_intensities)
            
            background = sorted_intensities[:n_pixels//4]
            signal = sorted_intensities[3*n_pixels//4:]
            
            mean_signal = np.mean(signal)
            std_background = np.std(background)
            
            snr = mean_signal / std_background if std_background > 0 else 0
            return float(snr)
        except:
            return 0.0
    
    def _calculate_rms_contrast(self, image: np.ndarray) -> float:
        """Calculate RMS contrast"""
        try:
            mean_intensity = np.mean(image)
            rms_contrast = np.sqrt(np.mean((image - mean_intensity) ** 2))
            return float(rms_contrast)
        except:
            return 0.0
    
    def _calculate_michelson_contrast(self, image: np.ndarray) -> float:
        """Calculate Michelson contrast"""
        try:
            max_intensity = np.max(image)
            min_intensity = np.min(image)
            
            if max_intensity + min_intensity > 0:
                contrast = (max_intensity - min_intensity) / (max_intensity + min_intensity)
            else:
                contrast = 0
            
            return float(contrast)
        except:
            return 0.0
    
    def _calculate_gradient_magnitude(self, image: np.ndarray) -> float:
        """Calculate average gradient magnitude (sharpness indicator)"""
        try:
            from skimage.filters import sobel
            gradient = sobel(image)
            return float(np.mean(gradient))
        except:
            # Fallback to simple gradient calculation
            gy, gx = np.gradient(image)
            gradient_magnitude = np.sqrt(gx**2 + gy**2)
            return float(np.mean(gradient_magnitude))
    
    def _calculate_laplacian_variance(self, image: np.ndarray) -> float:
        """Calculate Laplacian variance (focus measure)"""
        try:
            from scipy.ndimage import laplace
            laplacian = laplace(image)
            return float(np.var(laplacian))
        except:
            return 0.0
    
    def _estimate_noise_level(self, image: np.ndarray) -> float:
        """Estimate noise level using local standard deviation"""
        try:
            from scipy.ndimage import uniform_filter
            
            # Local mean
            local_mean = uniform_filter(image.astype(float), size=3)
            
            # Local variance
            local_var = uniform_filter(image.astype(float)**2, size=3) - local_mean**2
            
            # Noise estimate as median local standard deviation
            noise_level = np.median(np.sqrt(np.maximum(local_var, 0)))
            
            return float(noise_level)
        except:
            return 0.0
    
    def _calculate_saturation_percentage(self, image: np.ndarray) -> float:
        """Calculate percentage of saturated pixels"""
        try:
            if image.dtype == np.uint8:
                max_value = 255
            elif image.dtype == np.uint16:
                max_value = 65535
            else:
                max_value = np.max(image)
            
            saturated_pixels = np.sum(image >= max_value * 0.99)  # Consider 99% of max as saturated
            total_pixels = image.size
            
            return float(saturated_pixels / total_pixels * 100)
        except:
            return 0.0

class SegmentationQualityChecker:
    """Validates cell segmentation quality"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_segmentation(self, masks: np.ndarray, original_image: np.ndarray) -> Dict[str, Any]:
        """Comprehensive segmentation validation"""
        
        validation_results = {}
        
        try:
            # Basic segmentation statistics
            unique_labels = np.unique(masks)
            n_cells = len(unique_labels) - 1  # Exclude background (0)
            
            validation_results['n_cells_detected'] = n_cells
            validation_results['coverage_percentage'] = self._calculate_coverage_percentage(masks)
            
            # Cell size distribution
            cell_areas = self._calculate_cell_areas(masks)
            validation_results['cell_areas'] = cell_areas
            validation_results['mean_cell_area'] = float(np.mean(cell_areas)) if cell_areas else 0
            validation_results['std_cell_area'] = float(np.std(cell_areas)) if cell_areas else 0
            validation_results['cv_cell_area'] = validation_results['std_cell_area'] / validation_results['mean_cell_area'] if validation_results['mean_cell_area'] > 0 else 0
            
            # Morphological validation
            validation_results['morphology_metrics'] = self._analyze_cell_morphology(masks)
            
            # Segmentation boundary quality
            validation_results['boundary_quality'] = self._assess_boundary_quality(masks, original_image)
            
            # Over/under-segmentation detection
            validation_results['segmentation_issues'] = self._detect_segmentation_issues(masks, cell_areas)
            
        except Exception as e:
            self.logger.error(f"Error validating segmentation: {e}")
            validation_results['error'] = str(e)
        
        return validation_results
    
    def _calculate_coverage_percentage(self, masks: np.ndarray) -> float:
        """Calculate percentage of image covered by cells"""
        try:
            cell_pixels = np.sum(masks > 0)
            total_pixels = masks.size
            return float(cell_pixels / total_pixels * 100)
        except:
            return 0.0
    
    def _calculate_cell_areas(self, masks: np.ndarray) -> List[float]:
        """Calculate area of each segmented cell"""
        try:
            from skimage import measure
            
            regions = measure.regionprops(masks)
            areas = [region.area for region in regions]
            return areas
        except:
            return []
    
    def _analyze_cell_morphology(self, masks: np.ndarray) -> Dict[str, float]:
        """Analyze morphological properties of segmented cells"""
        
        morphology_metrics = {
            'mean_eccentricity': 0.0,
            'mean_solidity': 0.0,
            'mean_circularity': 0.0,
            'irregularity_score': 0.0
        }
        
        try:
            from skimage import measure
            
            regions = measure.regionprops(masks)
            
            if regions:
                eccentricities = [region.eccentricity for region in regions]
                solidities = [region.solidity for region in regions]
                
                # Calculate circularity: 4π*area/perimeter²
                circularities = []
                for region in regions:
                    if region.perimeter > 0:
                        circularity = 4 * np.pi * region.area / (region.perimeter ** 2)
                        circularities.append(min(circularity, 1.0))  # Cap at 1.0
                
                morphology_metrics['mean_eccentricity'] = float(np.mean(eccentricities))
                morphology_metrics['mean_solidity'] = float(np.mean(solidities))
                morphology_metrics['mean_circularity'] = float(np.mean(circularities)) if circularities else 0.0
                
                # Irregularity score based on variation in shape metrics
                morphology_metrics['irregularity_score'] = float(np.std(eccentricities) + np.std(solidities))
        
        except Exception as e:
            self.logger.debug(f"Error analyzing morphology: {e}")
        
        return morphology_metrics
    
    def _assess_boundary_quality(self, masks: np.ndarray, original_image: np.ndarray) -> Dict[str, float]:
        """Assess quality of segmentation boundaries"""
        
        boundary_metrics = {
            'boundary_contrast': 0.0,
            'boundary_smoothness': 0.0,
            'edge_alignment_score': 0.0
        }
        
        try:
            from skimage import segmentation, filters
            
            # Find boundaries
            boundaries = segmentation.find_boundaries(masks, mode='inner')
            boundary_coords = np.where(boundaries)
            
            if len(boundary_coords[0]) > 0:
                # Boundary contrast - difference between cell interior and boundary region
                boundary_intensities = original_image[boundary_coords]
                
                # Get nearby interior pixels
                from scipy.ndimage import binary_dilation
                dilated_boundaries = binary_dilation(boundaries, iterations=3)
                interior_mask = (masks > 0) & ~dilated_boundaries
                
                if np.sum(interior_mask) > 0:
                    interior_intensities = original_image[interior_mask]
                    boundary_metrics['boundary_contrast'] = float(np.mean(interior_intensities) - np.mean(boundary_intensities))
                
                # Edge alignment with image gradients
                gradient_magnitude = filters.sobel(original_image)
                boundary_gradient_strength = gradient_magnitude[boundary_coords]
                boundary_metrics['edge_alignment_score'] = float(np.mean(boundary_gradient_strength))
        
        except Exception as e:
            self.logger.debug(f"Error assessing boundary quality: {e}")
        
        return boundary_metrics
    
    def _detect_segmentation_issues(self, masks: np.ndarray, cell_areas: List[float]) -> Dict[str, Any]:
        """Detect common segmentation issues"""
        
        issues = {
            'oversegmentation_detected': False,
            'undersegmentation_detected': False,
            'size_outliers': [],
            'fragmented_cells': 0
        }
        
        try:
            if not cell_areas:
                return issues
            
            # Detect size outliers using IQR method
            q1, q3 = np.percentile(cell_areas, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = [area for area in cell_areas if area < lower_bound or area > upper_bound]
            issues['size_outliers'] = outliers
            
            # Over-segmentation: many very small cells
            very_small_cells = sum(1 for area in cell_areas if area < np.percentile(cell_areas, 10))
            if very_small_cells > len(cell_areas) * 0.2:  # More than 20% are very small
                issues['oversegmentation_detected'] = True
            
            # Under-segmentation: few very large cells
            very_large_cells = sum(1 for area in cell_areas if area > np.percentile(cell_areas, 90))
            if very_large_cells > len(cell_areas) * 0.1 and np.max(cell_areas) > np.mean(cell_areas) * 3:
                issues['undersegmentation_detected'] = True
            
            # Fragmented cells: very small objects that might be cell fragments
            issues['fragmented_cells'] = sum(1 for area in cell_areas if area < np.mean(cell_areas) * 0.1)
        
        except Exception as e:
            self.logger.debug(f"Error detecting segmentation issues: {e}")
        
        return issues

class DataQualityValidator:
    """Validates quantitative measurement data quality"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_measurement_data(self, data: pd.DataFrame, experiment_config: Dict[str, Any]) -> List[ValidationResult]:
        """Comprehensive validation of measurement data"""
        
        validation_results = []
        
        # Check data completeness
        validation_results.extend(self._check_data_completeness(data))
        
        # Check for outliers
        validation_results.extend(self._detect_outliers(data))
        
        # Validate measurement distributions
        validation_results.extend(self._validate_distributions(data))
        
        # Check for systematic biases
        validation_results.extend(self._check_systematic_biases(data))
        
        # Validate condition comparisons
        validation_results.extend(self._validate_condition_comparisons(data))
        
        # Check measurement consistency
        validation_results.extend(self._check_measurement_consistency(data))
        
        return validation_results
    
    def _check_data_completeness(self, data: pd.DataFrame) -> List[ValidationResult]:
        """Check for missing data and completeness"""
        
        results = []
        
        # Check for missing values
        missing_counts = data.isnull().sum()
        missing_percentage = (missing_counts / len(data)) * 100
        
        for column, missing_pct in missing_percentage.items():
            if missing_pct > 0:
                if missing_pct > 10:
                    status = 'fail'
                    message = f"High missing data rate in {column}: {missing_pct:.1f}%"
                elif missing_pct > 5:
                    status = 'warning'
                    message = f"Moderate missing data rate in {column}: {missing_pct:.1f}%"
                else:
                    status = 'warning'
                    message = f"Low missing data rate in {column}: {missing_pct:.1f}%"
                
                results.append(ValidationResult(
                    check_name="Data Completeness",
                    status=status,
                    message=message,
                    details={'column': column, 'missing_percentage': missing_pct}
                ))
        
        # Check sample sizes per condition
        if 'condition' in data.columns:
            condition_counts = data['condition'].value_counts()
            min_samples = condition_counts.min()
            
            if min_samples < 10:
                results.append(ValidationResult(
                    check_name="Sample Size",
                    status='warning',
                    message=f"Low sample size detected: minimum {min_samples} cells per condition",
                    details={'condition_counts': condition_counts.to_dict()}
                ))
            elif min_samples < 30:
                results.append(ValidationResult(
                    check_name="Sample Size",
                    status='pass',
                    message=f"Adequate sample size: minimum {min_samples} cells per condition",
                    details={'condition_counts': condition_counts.to_dict()}
                ))
        
        return results
    
    def _detect_outliers(self, data: pd.DataFrame) -> List[ValidationResult]:
        """Detect outliers in measurement data"""
        
        results = []
        
        # Check CTCF columns for outliers
        ctcf_columns = [col for col in data.columns if col.endswith('_ctcf')]
        
        for column in ctcf_columns:
            if column not in data.columns or data[column].isnull().all():
                continue
            
            # Use IQR method for outlier detection
            q1 = data[column].quantile(0.25)
            q3 = data[column].quantile(0.75)
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
            outlier_percentage = (len(outliers) / len(data)) * 100
            
            if outlier_percentage > 10:
                status = 'warning'
                message = f"High outlier rate in {column}: {outlier_percentage:.1f}%"
            elif outlier_percentage > 5:
                status = 'pass'
                message = f"Moderate outlier rate in {column}: {outlier_percentage:.1f}%"
            else:
                status = 'pass'
                message = f"Normal outlier rate in {column}: {outlier_percentage:.1f}%"
            
            results.append(ValidationResult(
                check_name="Outlier Detection",
                status=status,
                message=message,
                details={
                    'column': column,
                    'outlier_percentage': outlier_percentage,
                    'n_outliers': len(outliers),
                    'bounds': {'lower': lower_bound, 'upper': upper_bound}
                }
            ))
        
        return results
    
    def _validate_distributions(self, data: pd.DataFrame) -> List[ValidationResult]:
        """Validate measurement distributions"""
        
        results = []
        
        ctcf_columns = [col for col in data.columns if col.endswith('_ctcf')]
        
        for column in ctcf_columns:
            if column not in data.columns or data[column].isnull().all():
                continue
            
            values = data[column].dropna()
            
            # Check for normality
            if len(values) > 8:  # Minimum sample size for Shapiro-Wilk
                try:
                    statistic, p_value = stats.shapiro(values)
                    
                    if p_value > 0.05:
                        status = 'pass'
                        message = f"Data distribution in {column} appears normal (p={p_value:.3f})"
                    else:
                        status = 'warning'
                        message = f"Data distribution in {column} may not be normal (p={p_value:.3f})"
                    
                    results.append(ValidationResult(
                        check_name="Distribution Normality",
                        status=status,
                        message=message,
                        details={
                            'column': column,
                            'shapiro_statistic': statistic,
                            'p_value': p_value
                        }
                    ))
                
                except Exception as e:
                    self.logger.debug(f"Error in normality test for {column}: {e}")
            
            # Check for reasonable value ranges
            if np.any(values < 0):
                results.append(ValidationResult(
                    check_name="Value Range Check",
                    status='warning',
                    message=f"Negative CTCF values detected in {column}",
                    details={
                        'column': column,
                        'n_negative': np.sum(values < 0),
                        'min_value': float(np.min(values))
                    }
                ))
        
        return results
    
    def _check_systematic_biases(self, data: pd.DataFrame) -> List[ValidationResult]:
        """Check for systematic biases in the data"""
        
        results = []
        
        # Check for position effects (if position data available)
        if 'centroid_x' in data.columns and 'centroid_y' in data.columns:
            ctcf_columns = [col for col in data.columns if col.endswith('_ctcf')]
            
            for column in ctcf_columns[:1]:  # Check first CTCF column as example
                if column not in data.columns:
                    continue
                
                # Test correlation with position
                corr_x = data[column].corr(data['centroid_x'])
                corr_y = data[column].corr(data['centroid_y'])
                
                if abs(corr_x) > 0.3 or abs(corr_y) > 0.3:
                    results.append(ValidationResult(
                        check_name="Position Bias Check",
                        status='warning',
                        message=f"Potential position bias detected in {column} (r_x={corr_x:.3f}, r_y={corr_y:.3f})",
                        details={
                            'column': column,
                            'correlation_x': corr_x,
                            'correlation_y': corr_y
                        }
                    ))
        
        return results
    
    def _validate_condition_comparisons(self, data: pd.DataFrame) -> List[ValidationResult]:
        """Validate comparisons between experimental conditions"""
        
        results = []
        
        if 'condition' not in data.columns:
            return results
        
        conditions = data['condition'].unique()
        if len(conditions) < 2:
            return results
        
        ctcf_columns = [col for col in data.columns if col.endswith('_ctcf')]
        
        for column in ctcf_columns:
            if column not in data.columns:
                continue
            
            # Check effect sizes between conditions
            condition_means = data.groupby('condition')[column].mean()
            overall_std = data[column].std()
            
            max_diff = condition_means.max() - condition_means.min()
            effect_size = max_diff / overall_std if overall_std > 0 else 0
            
            if effect_size > 0.8:
                status = 'pass'
                message = f"Large effect size detected in {column} (Cohen's d ≈ {effect_size:.2f})"
            elif effect_size > 0.5:
                status = 'pass'
                message = f"Medium effect size detected in {column} (Cohen's d ≈ {effect_size:.2f})"
            elif effect_size > 0.2:
                status = 'pass'
                message = f"Small effect size detected in {column} (Cohen's d ≈ {effect_size:.2f})"
            else:
                status = 'warning'
                message = f"Very small effect size in {column} (Cohen's d ≈ {effect_size:.2f})"
            
            results.append(ValidationResult(
                check_name="Effect Size Analysis",
                status=status,
                message=message,
                details={
                    'column': column,
                    'effect_size': effect_size,
                    'condition_means': condition_means.to_dict()
                }
            ))
        
        return results
    
    def _check_measurement_consistency(self, data: pd.DataFrame) -> List[ValidationResult]:
        """Check consistency of measurements"""
        
        results = []
        
        # Check if area measurements are reasonable
        if 'area' in data.columns:
            areas = data['area'].dropna()
            
            if len(areas) > 0:
                cv_area = areas.std() / areas.mean() if areas.mean() > 0 else 0
                
                if cv_area > 1.0:
                    status = 'warning'
                    message = f"High variability in cell areas (CV = {cv_area:.2f})"
                elif cv_area > 0.5:
                    status = 'pass'
                    message = f"Moderate variability in cell areas (CV = {cv_area:.2f})"
                else:
                    status = 'pass'
                    message = f"Low variability in cell areas (CV = {cv_area:.2f})"
                
                results.append(ValidationResult(
                    check_name="Measurement Consistency",
                    status=status,
                    message=message,
                    details={'area_cv': cv_area}
                ))
        
        return results

class QualityControlManager:
    """Main quality control manager that coordinates all validation checks"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.image_analyzer = ImageQualityAnalyzer()
        self.segmentation_checker = SegmentationQualityChecker()
        self.data_validator = DataQualityValidator()
    
    def generate_quality_report(self, experiment_results: Dict[str, Any], 
                              experiment_config: Dict[str, Any]) -> QualityReport:
        """Generate comprehensive quality control report"""
        
        self.logger.info("Generating quality control report")
        
        metrics = []
        validation_results = []
        recommendations = []
        
        # Analyze each condition
        for condition_name, condition_data in experiment_results.get('conditions', {}).items():
            
            # Image quality analysis (if image data available)
            if 'images' in condition_data:
                for image_info in condition_data['images']:
                    # This would need image data to be preserved or re-loaded
                    pass
            
            # Segmentation quality validation
            # This would need mask data to be preserved
            
            # Data quality validation
            if 'cell_data' in condition_data and condition_data['cell_data']:
                df = pd.DataFrame(condition_data['cell_data'])
                data_validations = self.data_validator.validate_measurement_data(df, experiment_config)
                validation_results.extend(data_validations)
        
        # Generate overall metrics
        overall_metrics = self._generate_overall_metrics(experiment_results)
        metrics.extend(overall_metrics)
        
        # Generate recommendations based on validation results
        recommendations = self._generate_recommendations(validation_results, metrics)
        
        # Determine overall status
        overall_status = self._determine_overall_status(validation_results, metrics)
        
        # Create data summary
        data_summary = self._create_data_summary(experiment_results)
        
        report = QualityReport(
            experiment_name=experiment_config.get('experiment_name', 'Unknown'),
            timestamp=datetime.now(),
            overall_status=overall_status,
            metrics=metrics,
            validation_results=validation_results,
            recommendations=recommendations,
            data_summary=data_summary
        )
        
        self.logger.info(f"Quality report generated with {len(validation_results)} validation checks")
        return report
    
    def _generate_overall_metrics(self, experiment_results: Dict[str, Any]) -> List[QualityMetric]:
        """Generate overall quality metrics"""
        
        metrics = []
        
        # Overall cell count metric
        total_cells = sum(len(cond_data.get('cell_data', [])) 
                         for cond_data in experiment_results.get('conditions', {}).values())
        
        if total_cells > 500:
            status = 'pass'
        elif total_cells > 100:
            status = 'warning'
        else:
            status = 'fail'
        
        metrics.append(QualityMetric(
            name="Total Cell Count",
            value=total_cells,
            threshold=100,
            status=status,
            description="Total number of cells analyzed across all conditions",
            recommendation="Aim for >100 cells per condition for robust statistics"
        ))
        
        # Condition balance metric
        conditions = experiment_results.get('conditions', {})
        if len(conditions) > 1:
            cell_counts = [len(cond_data.get('cell_data', [])) for cond_data in conditions.values()]
            cv_balance = np.std(cell_counts) / np.mean(cell_counts) if np.mean(cell_counts) > 0 else 0
            
            if cv_balance < 0.2:
                status = 'pass'
            elif cv_balance < 0.5:
                status = 'warning'
            else:
                status = 'fail'
            
            metrics.append(QualityMetric(
                name="Condition Balance",
                value=cv_balance,
                threshold=0.2,
                status=status,
                description="Balance of cell counts between conditions (CV)",
                recommendation="Keep cell counts similar between conditions for valid comparisons"
            ))
        
        return metrics
    
    def _generate_recommendations(self, validation_results: List[ValidationResult], 
                                metrics: List[QualityMetric]) -> List[str]:
        """Generate recommendations based on validation results"""
        
        recommendations = []
        
        # Count issues by type
        warnings = sum(1 for result in validation_results if result.status == 'warning')
        failures = sum(1 for result in validation_results if result.status == 'fail')
        
        if failures > 0:
            recommendations.append(f"Address {failures} critical validation failures before proceeding")
        
        if warnings > 3:
            recommendations.append(f"Review {warnings} validation warnings for potential data quality issues")
        
        # Specific recommendations based on common issues
        outlier_issues = [r for r in validation_results if 'outlier' in r.message.lower()]
        if len(outlier_issues) > 2:
            recommendations.append("High outlier rates detected - consider reviewing segmentation parameters")
        
        sample_size_issues = [r for r in validation_results if 'sample size' in r.message.lower()]
        if sample_size_issues:
            recommendations.append("Increase sample size for more robust statistical analysis")
        
        position_bias_issues = [r for r in validation_results if 'position bias' in r.message.lower()]
        if position_bias_issues:
            recommendations.append("Position-dependent effects detected - check for imaging artifacts")
        
        # Add metric-based recommendations
        for metric in metrics:
            if metric.status == 'fail' and metric.recommendation:
                recommendations.append(metric.recommendation)
        
        return recommendations
    
    def _determine_overall_status(self, validation_results: List[ValidationResult], 
                                metrics: List[QualityMetric]) -> str:
        """Determine overall quality status"""
        
        failures = sum(1 for result in validation_results if result.status == 'fail')
        warnings = sum(1 for result in validation_results if result.status == 'warning')
        
        metric_failures = sum(1 for metric in metrics if metric.status == 'fail')
        metric_warnings = sum(1 for metric in metrics if metric.status == 'warning')
        
        total_failures = failures + metric_failures
        total_warnings = warnings + metric_warnings
        
        if total_failures > 0:
            return 'fail'
        elif total_warnings > 3:
            return 'warning'
        else:
            return 'pass'
    
    def _create_data_summary(self, experiment_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of analyzed data"""
        
        summary = {
            'total_conditions': len(experiment_results.get('conditions', {})),
            'total_images': 0,
            'total_cells': 0,
            'conditions': {}
        }
        
        for condition_name, condition_data in experiment_results.get('conditions', {}).items():
            n_images = len(condition_data.get('images', []))
            n_cells = len(condition_data.get('cell_data', []))
            
            summary['total_images'] += n_images
            summary['total_cells'] += n_cells
            
            summary['conditions'][condition_name] = {
                'n_images': n_images,
                'n_cells': n_cells
            }
        
        return summary
    
    def save_quality_report(self, report: QualityReport, output_file: Path):
        """Save quality report to file"""
        
        report_data = {
            'experiment_name': report.experiment_name,
            'timestamp': report.timestamp.isoformat(),
            'overall_status': report.overall_status,
            'metrics': [
                {
                    'name': m.name,
                    'value': m.value,
                    'threshold': m.threshold,
                    'status': m.status,
                    'description': m.description,
                    'recommendation': m.recommendation
                }
                for m in report.metrics
            ],
            'validation_results': [
                {
                    'check_name': v.check_name,
                    'status': v.status,
                    'message': v.message,
                    'details': v.details,
                    'timestamp': v.timestamp.isoformat()
                }
                for v in report.validation_results
            ],
            'recommendations': report.recommendations,
            'data_summary': report.data_summary
        }
        
        with open(output_file, 'w') as f:
            import json
            json.dump(report_data, f, indent=2, default=str)
        
        self.logger.info(f"Quality report saved to: {output_file}")
    
    def create_quality_visualization(self, report: QualityReport, output_dir: Path):
        """Create visualizations for quality report"""
        
        output_dir.mkdir(exist_ok=True)
        
        # Status summary pie chart
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Quality Control Report Summary', fontsize=16, fontweight='bold')
        
        # Validation results summary
        status_counts = defaultdict(int)
        for result in report.validation_results:
            status_counts[result.status] += 1
        
        if status_counts:
            labels = list(status_counts.keys())
            sizes = list(status_counts.values())
            colors = {'pass': 'green', 'warning': 'orange', 'fail': 'red'}
            plot_colors = [colors.get(label, 'gray') for label in labels]
            
            ax1.pie(sizes, labels=labels, colors=plot_colors, autopct='%1.1f%%', startangle=90)
            ax1.set_title('Validation Results')
        
        # Metrics status
        metric_status_counts = defaultdict(int)
        for metric in report.metrics:
            metric_status_counts[metric.status] += 1
        
        if metric_status_counts:
            labels = list(metric_status_counts.keys())
            sizes = list(metric_status_counts.values())
            plot_colors = [colors.get(label, 'gray') for label in labels]
            
            ax2.pie(sizes, labels=labels, colors=plot_colors, autopct='%1.1f%%', startangle=90)
            ax2.set_title('Metrics Status')
        
        # Data summary bar chart
        if report.data_summary.get('conditions'):
            condition_names = list(report.data_summary['conditions'].keys())
            cell_counts = [report.data_summary['conditions'][name]['n_cells'] for name in condition_names]
            
            ax3.bar(condition_names, cell_counts, alpha=0.7)
            ax3.set_title('Cell Counts by Condition')
            ax3.set_ylabel('Number of Cells')
            ax3.tick_params(axis='x', rotation=45)
        
        # Overall status indicator
        status_color = {'pass': 'green', 'warning': 'orange', 'fail': 'red'}.get(report.overall_status, 'gray')
        ax4.text(0.5, 0.5, f'Overall Status:\n{report.overall_status.upper()}', 
                ha='center', va='center', fontsize=24, fontweight='bold', color=status_color,
                transform=ax4.transAxes)
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = output_dir / "quality_summary.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Quality visualization saved to: {plot_file}")

# Example usage
def demonstrate_quality_control():
    """Demonstrate quality control capabilities"""
    
    print("CellQuantGUI Quality Control Demo")
    print("=" * 40)
    
    # Initialize quality control manager
    qc_manager = QualityControlManager()
    
    # Example experiment results (normally would come from analysis pipeline)
    mock_results = {
        'conditions': {
            'Control': {
                'cell_data': [
                    {'cell_id': i, 'area': 100 + np.random.normal(0, 20), 
                     'target_protein_ctcf': 1000 + np.random.normal(0, 200)}
                    for i in range(50)
                ],
                'images': ['image1.tif', 'image2.tif']
            },
            'Treatment': {
                'cell_data': [
                    {'cell_id': i, 'area': 120 + np.random.normal(0, 25), 
                     'target_protein_ctcf': 1500 + np.random.normal(0, 300)}
                    for i in range(45)
                ],
                'images': ['image3.tif', 'image4.tif']
            }
        }
    }
    
    mock_config = {
        'experiment_name': 'Demo_Experiment',
        'conditions': [
            {'name': 'Control', 'channels': []},
            {'name': 'Treatment', 'channels': []}
        ]
    }
    
    try:
        # Generate quality report
        report = qc_manager.generate_quality_report(mock_results, mock_config)
        
        print(f"\nQuality Report Generated:")
        print(f"Overall Status: {report.overall_status}")
        print(f"Validation Checks: {len(report.validation_results)}")
        print(f"Quality Metrics: {len(report.metrics)}")
        print(f"Recommendations: {len(report.recommendations)}")
        
        # Show some validation results
        print("\nValidation Results:")
        for result in report.validation_results[:3]:
            print(f"  - {result.check_name}: {result.status} - {result.message}")
        
        # Show recommendations
        if report.recommendations:
            print(f"\nRecommendations:")
            for rec in report.recommendations[:3]:
                print(f"  - {rec}")
    
    except Exception as e:
        print(f"Demo error: {e}")

if __name__ == "__main__":
    demonstrate_quality_control()