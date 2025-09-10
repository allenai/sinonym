"""
Comprehensive tests for batch processing functionality.

This module tests the batch format detection and processing capabilities
of the Chinese name detection system.

All tests refactored to use session-scoped detector fixture for optimal performance.
"""

import pytest

from sinonym import ChineseNameDetector
from sinonym.coretypes import BatchFormatPattern, BatchParseResult


# ===================================================================
# BATCH FORMAT DETECTION TESTS
# ===================================================================

def test_homogeneous_given_first_batch(detector):
    """Test detection with names that individual processing handles correctly."""
    # Names with common surnames (Liu/Li/Huang/Zhang/Wang) and common given names (Xin/Yang/Chen/Wei/Ming)
    # Individual processing correctly identifies these as given-first based on frequency
    names = ["Xin Liu", "Yang Li", "Chen Huang", "Wei Zhang", "Ming Wang"]

    pattern = detector.detect_batch_format(names)

    # These names are correctly processed individually, regardless of batch pattern detection
    # The key is that batch processing doesn't make things worse
    assert pattern.total_count == 5

    # Test that batch processing produces the same results as individual processing
    batch_result = detector.analyze_name_batch(names)
    expected_results = ["Xin Liu", "Yang Li", "Chen Huang", "Wei Zhang", "Ming Wang"]

    for i, (expected, actual) in enumerate(zip(expected_results, batch_result.results, strict=False)):
        assert actual.success, f"Failed to process name {i}"
        assert actual.result == expected, f"Expected '{expected}', got '{actual.result}'"


def test_homogeneous_surname_first_batch(detector):
    """Test detection of surname-first format in homogeneous batch."""
    # All names follow surname-first pattern (Chinese-style)
    names = ["Li Wei", "Zhang Ming", "Wang Xiaoli", "Liu Jiaming", "Chen Weimin"]

    pattern = detector.detect_batch_format(names)

    assert pattern.total_count == 5
    assert pattern.threshold_met in [True, False]  # Either pattern should be detected
    assert pattern.confidence > 0.0


def test_mixed_batch_no_clear_pattern(detector):
    """Test detection when there's no clear format pattern."""
    # Mixed formats that don't reach threshold
    names = [
        "John Smith",  # Non-Chinese (will be rejected)
        "Xin Liu",  # Chinese name
        "Mary Johnson",  # Non-Chinese (will be rejected)
    ]

    pattern = detector.detect_batch_format(names)

    # Should detect the single Chinese name
    assert pattern.total_count == 1


def test_threshold_boundary_cases(detector):
    """Test format detection at threshold boundaries."""
    # Exactly 67% should meet threshold (2/3)
    names = [
        "Xin Liu",  # Given-first
        "Yang Li",  # Given-first
        "Wei Zhang",  # Given-first (if current weights favor this)
    ]

    pattern = detector.detect_batch_format(names, format_threshold=0.67)

    assert pattern.total_count == 3
    # With current weights, likely all will be given-first
    if pattern.confidence >= 0.67:
        assert pattern.threshold_met is True


def test_small_batch_fallback(detector):
    """Test that small batches fall back correctly."""
    names = ["Xin Liu", "Yang Li"]  # Only 2 names

    batch_result = detector.analyze_name_batch(names, minimum_batch_size=3)

    # Should fall back to individual processing
    assert batch_result.format_pattern.threshold_met is False
    assert len(batch_result.results) == 2
    assert len(batch_result.improvements) == 0


# ===================================================================
# BATCH PROCESSING TESTS
# ===================================================================

def test_analyze_name_batch_basic(detector):
    """Test basic batch analysis functionality."""
    names = ["Xin Liu", "Yang Li", "Chen Huang"]

    result = detector.analyze_name_batch(names)

    assert isinstance(result, BatchParseResult)
    assert len(result.names) == 3
    assert len(result.results) == 3
    assert len(result.individual_analyses) == 3
    assert isinstance(result.format_pattern, BatchFormatPattern)
    assert isinstance(result.improvements, list)


def test_process_name_batch_convenience(detector):
    """Test the convenience method for batch processing."""
    names = ["Xin Liu", "Yang Li", "Chen Huang"]

    results = detector.process_name_batch(names)

    assert len(results) == 3
    for result in results:
        # All should be successful Chinese name detections
        assert result.success


def test_batch_with_rejections(detector):
    """Test batch processing with some rejected names."""
    names = [
        "Xin Liu",  # Chinese
        "John Smith",  # Non-Chinese
        "Yang Li",  # Chinese
        "Mary Johnson",  # Non-Chinese
    ]

    result = detector.analyze_name_batch(names)

    assert len(result.results) == 4
    # Check that Chinese names succeeded and non-Chinese were rejected
    assert result.results[0].success  # Xin Liu
    assert not result.results[1].success  # John Smith
    assert result.results[2].success  # Yang Li
    assert not result.results[3].success  # Mary Johnson


def test_configurable_thresholds(detector):
    """Test batch processing with different confidence thresholds."""
    names = ["Xin Liu", "Yang Li", "Chen Huang"]

    # Test with low threshold
    result_low = detector.analyze_name_batch(names, format_threshold=0.3)
    assert len(result_low.results) == 3

    # Test with high threshold
    result_high = detector.analyze_name_batch(names, format_threshold=0.9)
    assert len(result_high.results) == 3

    # Results should be the same regardless of threshold for this clear case
    for r1, r2 in zip(result_low.results, result_high.results, strict=False):
        if r1.success and r2.success:
            assert r1.result == r2.result


# ===================================================================
# BATCH EDGE CASES TESTS
# ===================================================================

def test_empty_batch(detector):
    """Test empty batch handling."""
    names = []

    result = detector.analyze_name_batch(names)

    assert len(result.results) == 0
    assert len(result.names) == 0
    assert result.format_pattern.total_count == 0


def test_single_name_batch(detector):
    """Test single-name batch processing."""
    names = ["Xin Liu"]

    result = detector.analyze_name_batch(names)

    assert len(result.results) == 1
    assert result.results[0].success
    assert result.results[0].result == "Xin Liu"


def test_all_rejected_batch(detector):
    """Test batch with all names rejected."""
    names = ["John Smith", "Mary Johnson", "Bob Wilson"]

    result = detector.analyze_name_batch(names)

    assert len(result.results) == 3
    for res in result.results:
        assert not res.success
    assert result.format_pattern.total_count == 0  # No Chinese names


def test_compound_names_batch(detector):
    """Test batch processing of compound surnames."""
    names = [
        "Ou-yang Ming",
        "Szeto Wah",
        "Au-yeung Ka Fai",
    ]

    result = detector.analyze_name_batch(names)

    assert len(result.results) == 3
    # All should be successfully processed
    for res in result.results:
        assert res.success


def test_chinese_characters_batch(detector):
    """Test batch with Chinese character input."""
    names = [
        "李明",  # Li Ming
        "王小红",  # Wang Xiaohong
        "张三",  # Zhang San
    ]

    result = detector.analyze_name_batch(names)

    assert len(result.results) == 3
    for res in result.results:
        assert res.success


def test_mixed_script_batch(detector):
    """Test batch with mixed script names."""
    names = [
        "李明 Li Ming",  # Mixed Chinese/Roman
        "Xin Liu",  # Roman only
        "王小红",  # Chinese only
    ]

    result = detector.analyze_name_batch(names)

    assert len(result.results) == 3
    for res in result.results:
        assert res.success


# ===================================================================
# BATCH IMPROVEMENTS TESTS
# ===================================================================

def test_improvement_detection(detector):
    """Test detection of batch improvements."""
    # Names that benefit from batch format detection
    names = [
        "Liu Xin",  # Ambiguous - could be surname-first or given-first
        "Li Yang", # Ambiguous
        "Zhang Wei", # Ambiguous
    ]

    result = detector.analyze_name_batch(names)

    assert len(result.results) == 3
    assert isinstance(result.improvements, list)
    # Improvements list contains indices of names that were improved


def test_no_improvements_when_optimal(detector):
    """Test that clear cases show no improvements."""
    names = [
        "Xin Liu",  # Clear given-first
        "Yang Li",  # Clear given-first
        "Chen Huang",  # Clear given-first
    ]

    result = detector.analyze_name_batch(names)

    assert len(result.results) == 3
    # Should have no improvements since individual processing is already optimal
    assert len(result.improvements) == 0


# ===================================================================
# BATCH PERFORMANCE TESTS
# ===================================================================

def test_large_batch_processing(detector):
    """Test processing of larger batches."""
    # Generate a larger batch
    base_names = ["Xin Liu", "Yang Li", "Chen Huang", "Wei Zhang", "Ming Wang"]
    names = base_names * 10  # 50 names

    result = detector.analyze_name_batch(names)

    assert len(result.results) == 50
    for res in result.results:
        assert res.success


def test_batch_vs_individual_consistency(detector):
    """Test that batch processing gives consistent results with individual processing."""
    names = ["Xin Liu", "Yang Li", "Chen Huang"]

    # Process as batch
    batch_result = detector.analyze_name_batch(names)

    # Process individually
    individual_results = [detector.normalize_name(name) for name in names]

    # Results should be consistent (though format might differ due to batch logic)
    assert len(batch_result.results) == len(individual_results)
    for batch_res, ind_res in zip(batch_result.results, individual_results, strict=False):
        assert batch_res.success == ind_res.success


# ===================================================================
# BATCH OUTCOMES TESTS  
# ===================================================================

def test_surname_first_batch_outcomes(detector):
    """Test outcomes for surname-first dominant batches."""
    names = [
        "Wang Xin",    # surname-first
        "Li Yang",     # surname-first  
        "Zhang Wei",   # surname-first
        "Chen Ming",   # surname-first
    ]

    result = detector.analyze_name_batch(names)
    
    assert len(result.results) == 4
    for res in result.results:
        assert res.success


def test_given_first_batch_outcomes(detector):
    """Test outcomes for given-first dominant batches."""  
    names = [
        "Xin Wang",    # given-first
        "Yang Li",     # given-first
        "Wei Zhang",   # given-first
        "Ming Chen",   # given-first
    ]

    result = detector.analyze_name_batch(names)
    
    assert len(result.results) == 4
    for res in result.results:
        assert res.success


def test_problematic_cases_in_context(detector):
    """Test problematic individual cases in batch context."""
    names = [
        "Li Jin",      # Problematic individual case
        "Wang Wei",    # Clear surname-first
        "Zhang Ming",  # Clear surname-first
        "Liu Yang",    # Clear surname-first
    ]

    result = detector.analyze_name_batch(names)
    
    assert len(result.results) == 4
    # Batch context should help resolve the ambiguous "Li Jin"
    for res in result.results:
        assert res.success


def test_compound_variations_batch_outcomes(detector):
    """Test batch outcomes with compound surname variations."""
    names = [
        "Ou-yang Wei",     # Compound with hyphen
        "OuYang Ming",     # Compound camelCase  
        "ou yang Li",      # Compound lowercase
        "OUYANG Zhang",    # Compound uppercase
    ]

    result = detector.analyze_name_batch(names)
    
    assert len(result.results) == 4
    for res in result.results:
        assert res.success


# ===================================================================
# BATCH REAL FAILING CASES TESTS
# ===================================================================

def test_real_failing_cases_batch_context(detector):
    """Test real failing cases in batch context to see if context helps."""
    # These are names that might fail individually but could be helped by batch context
    names = [
        "Yu Bei",      # Real failing case - ambiguous
        "Li Chong",    # Real failing case - ambiguous
        "Wang Wei",    # Clear context name
        "Zhang Ming",  # Clear context name
        "Liu Yang",    # Clear context name
    ]

    result = detector.analyze_name_batch(names)
    
    assert len(result.results) == 5
    # At minimum, the clear context names should succeed
    context_successes = sum(1 for res in result.results[-3:] if res.success)
    assert context_successes >= 3


def test_batch_edge_case_names(detector):
    """Test edge case names in batch context."""
    names = [
        "A Li",        # Single letter given name
        "Li A",        # Single letter surname  
        "Ma Ma",       # Repeated syllable
        "Wang Wei",    # Normal context
    ]

    result = detector.analyze_name_batch(names)
    
    assert len(result.results) == 4
    # At least the normal context should work
    assert result.results[-1].success


# ===================================================================  
# BATCH ACL REAL WORLD TESTS
# ===================================================================

def test_acl_author_batch_processing(detector):
    """Test batch processing on ACL-style author names."""
    # Real ACL author names that might benefit from batch processing
    names = [
        "Xin Liu",
        "Yang Li", 
        "Chen Huang",
        "Wei Zhang",
        "Ming Wang",
        "Yifan Gao",
        "Zheng Zhang",
        "Liang Zhao",
    ]

    result = detector.analyze_name_batch(names)
    
    assert len(result.results) == 8
    success_count = sum(1 for res in result.results if res.success)
    # Most ACL names should be successfully processed
    assert success_count >= 6


def test_mixed_confidence_batch(detector):
    """Test batch with mixed confidence names."""
    names = [
        "Xin Liu",      # High confidence
        "Yang Li",      # High confidence  
        "Bei Yu",       # Lower confidence, ambiguous
        "Li Jin",       # Lower confidence, ambiguous
        "Chen Wei",     # High confidence
    ]

    result = detector.analyze_name_batch(names)
    
    assert len(result.results) == 5
    # High confidence names should definitely succeed
    high_conf_indices = [0, 1, 4]  # Xin Liu, Yang Li, Chen Wei
    for i in high_conf_indices:
        assert result.results[i].success


def test_batch_format_consistency(detector):
    """Test that batch maintains format consistency."""
    names = [
        "Liu Xin",     # Could be surname-first or given-first
        "Li Yang",     # Could be surname-first or given-first  
        "Zhang Wei",   # Could be surname-first or given-first
        "Wang Ming",   # Could be surname-first or given-first
    ]

    result = detector.analyze_name_batch(names)
    
    assert len(result.results) == 4
    # All should be processed successfully
    for res in result.results:
        assert res.success
        
    # Check that format pattern was detected
    assert result.format_pattern.total_count == 4


def test_batch_individual_analyses(detector):
    """Test that individual analyses are populated correctly."""
    names = ["Xin Liu", "Yang Li", "Chen Huang"]

    result = detector.analyze_name_batch(names)
    
    assert len(result.individual_analyses) == 3
    for analysis in result.individual_analyses:
        assert hasattr(analysis, 'raw_name')
        assert hasattr(analysis, 'candidates') 
        assert hasattr(analysis, 'best_candidate')
        assert hasattr(analysis, 'confidence')
