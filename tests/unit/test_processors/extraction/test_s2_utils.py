# tests/unit/test_utils/test_s2_utils.py
"""
Tests for S2 utilities using s2sphere.

Tests the Windows-compatible S2 geometry utilities that replace
the Google Colab s2geometry library.
"""

import pytest

try:
    from shapely.geometry import box, Polygon, Point
    from shapely.ops import unary_union
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False

pytestmark = pytest.mark.skipif(
    not HAS_SHAPELY,
    reason="Shapely not available"
)

from geoworkflow.utils.s2_utils import (
    get_bounding_box_s2_covering_tokens,
    s2_token_to_shapely_polygon,
    estimate_cells_for_geometry,
    validate_s2_level,
    get_s2_cell_area
)


class TestS2TokenGeneration:
    """Test S2 token generation for geometries."""
    
    def test_token_generation_for_simple_bbox(self):
        """Test S2 token generation for simple bounding box."""
        # Create a simple 1x1 degree box
        bbox = box(-1, 5, 0, 6)
        
        tokens = get_bounding_box_s2_covering_tokens(bbox, level=6)
        
        # Should generate at least one token
        assert len(tokens) > 0
        
        # All tokens should be strings
        assert all(isinstance(t, str) for t in tokens)
        
        # Tokens should be reasonable length (S2 tokens are typically 16 chars at level 6)
        assert all(len(t) > 0 for t in tokens)
    
    def test_token_generation_for_large_area(self):
        """Test token generation for large geographic area."""
        # Large area (roughly country-sized)
        large_box = box(-10, 0, 10, 20)
        
        tokens = get_bounding_box_s2_covering_tokens(large_box, level=6)
        
        # Should generate many tokens
        assert len(tokens) > 10
        
        # Should not exceed reasonable limits
        assert len(tokens) < 10000
    
    def test_token_generation_for_small_area(self):
        """Test token generation for small area (city-sized)."""
        # Small area (roughly city-sized)
        small_box = box(-0.1, 5.5, -0.05, 5.55)
        
        tokens = get_bounding_box_s2_covering_tokens(small_box, level=6)
        
        # Should generate few tokens
        assert len(tokens) >= 1
        assert len(tokens) < 10
    
    def test_token_generation_different_levels(self):
        """Test that different S2 levels produce different numbers of tokens."""
        bbox = box(-1, 5, 0, 6)
        
        tokens_level_4 = get_bounding_box_s2_covering_tokens(bbox, level=4)
        tokens_level_6 = get_bounding_box_s2_covering_tokens(bbox, level=6)
        tokens_level_8 = get_bounding_box_s2_covering_tokens(bbox, level=8)
        
        # Higher levels should produce more tokens (finer coverage)
        assert len(tokens_level_4) < len(tokens_level_6)
        assert len(tokens_level_6) < len(tokens_level_8)
    
    def test_token_generation_for_polygon(self):
        """Test token generation for complex polygon."""
        # Create irregular polygon
        coords = [(0, 0), (2, 0), (2, 1), (1, 2), (0, 1), (0, 0)]
        polygon = Polygon(coords)
        
        tokens = get_bounding_box_s2_covering_tokens(polygon, level=6)
        
        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)
    
    def test_token_generation_max_cells_limit(self):
        """Test that max_cells parameter limits token count."""
        large_box = box(-50, -50, 50, 50)
        
        tokens = get_bounding_box_s2_covering_tokens(
            large_box, 
            level=6, 
            max_cells=100
        )
        
        # Should respect max_cells limit
        assert len(tokens) <= 100
    
    def test_token_uniqueness(self):
        """Test that generated tokens are unique."""
        bbox = box(-5, 0, 5, 10)
        
        tokens = get_bounding_box_s2_covering_tokens(bbox, level=6)
        
        # All tokens should be unique
        assert len(tokens) == len(set(tokens))
    
    def test_token_format_validity(self):
        """Test that generated tokens have valid S2 format."""
        bbox = box(0, 0, 1, 1)
        
        tokens = get_bounding_box_s2_covering_tokens(bbox, level=6)
        
        for token in tokens:
            # S2 tokens should be alphanumeric
            assert token.replace('-', '').replace('_', '').isalnum()
            # Should not be empty
            assert len(token) > 0


class TestS2TokenToPolygon:
    """Test conversion of S2 tokens to Shapely polygons."""
    
    def test_token_to_polygon_conversion(self):
        """Test converting S2 token to polygon."""
        # First generate a token
        bbox = box(0, 0, 1, 1)
        tokens = get_bounding_box_s2_covering_tokens(bbox, level=6)
        
        # Convert first token back to polygon
        polygon = s2_token_to_shapely_polygon(tokens[0])
        
        # Should be a valid polygon
        assert polygon.is_valid
        assert polygon.geom_type == 'Polygon'
        
        # Should have 4 corners + closing point (5 coords)
        assert len(polygon.exterior.coords) == 5
    
    def test_polygon_geographic_bounds(self):
        """Test that converted polygon has reasonable geographic bounds."""
        bbox = box(-10, 40, -5, 45)
        tokens = get_bounding_box_s2_covering_tokens(bbox, level=6)
        
        for token in tokens[:3]:  # Test first few
            polygon = s2_token_to_shapely_polygon(token)
            
            # Should be within reasonable lat/lon bounds
            bounds = polygon.bounds
            assert -180 <= bounds[0] <= 180  # min lon
            assert -90 <= bounds[1] <= 90    # min lat
            assert -180 <= bounds[2] <= 180  # max lon
            assert -90 <= bounds[3] <= 90    # max lat
    
    def test_multiple_tokens_coverage(self):
        """Test that multiple S2 polygons reasonably cover input area."""
        bbox = box(0, 0, 2, 2)
        tokens = get_bounding_box_s2_covering_tokens(bbox, level=6)
        
        # Convert all tokens to polygons
        polygons = [s2_token_to_shapely_polygon(t) for t in tokens]
        
        # Union of all polygons should overlap with original bbox
        union = unary_union(polygons)
        
        # Union should intersect significantly with original
        intersection = union.intersection(bbox)
        assert intersection.area > 0


class TestEstimateCells:
    """Test S2 cell estimation functions."""
    
    def test_estimate_cells_for_geometry(self):
        """Test cell count estimation."""
        bbox = box(-1, 5, 0, 6)
        
        estimate = estimate_cells_for_geometry(bbox, level=6)
        
        # Should return a positive integer
        assert isinstance(estimate, int)
        assert estimate > 0
    
    def test_estimate_scales_with_area(self):
        """Test that estimate increases with geometry area."""
        small_box = box(0, 0, 1, 1)
        large_box = box(0, 0, 10, 10)
        
        small_estimate = estimate_cells_for_geometry(small_box, level=6)
        large_estimate = estimate_cells_for_geometry(large_box, level=6)
        
        # Larger area should require more cells
        assert large_estimate > small_estimate
    
    def test_estimate_vs_actual_tokens(self):
        """Test that estimate is reasonably close to actual token count."""
        bbox = box(0, 0, 5, 5)
        
        estimate = estimate_cells_for_geometry(bbox, level=6)
        actual_tokens = get_bounding_box_s2_covering_tokens(bbox, level=6)
        
        # Estimate should be within reasonable range of actual
        # (typically within 2x due to boundary effects)
        assert estimate > 0
        assert actual_tokens is not None
        ratio = len(actual_tokens) / estimate
        assert 0.5 < ratio < 2.0


class TestS2LevelValidation:
    """Test S2 level validation."""
    
    def test_validate_valid_levels(self):
        """Test validation accepts valid levels."""
        valid_levels = [4, 5, 6, 7, 8]
        
        for level in valid_levels:
            # Should not raise exception
            validate_s2_level(level)
    
    def test_validate_invalid_levels(self):
        """Test validation rejects invalid levels."""
        invalid_levels = [0, 3, 9, 15, -1, 31]
        
        for level in invalid_levels:
            with pytest.raises(ValueError, match="S2 level"):
                validate_s2_level(level)
    
    def test_validate_level_6_recommended(self):
        """Test that level 6 is the recommended default."""
        # Level 6 should validate without warnings
        validate_s2_level(6)


class TestS2CellArea:
    """Test S2 cell area calculations."""
    
    def test_get_cell_area_level_6(self):
        """Test cell area calculation for level 6."""
        area = get_s2_cell_area(level=6)
        
        # Should return positive area in km²
        assert area > 0
        
        # Level 6 cells are roughly 100-150 km² at equator
        assert 50 < area < 300
    
    def test_area_decreases_with_level(self):
        """Test that cell area decreases as level increases."""
        area_4 = get_s2_cell_area(level=4)
        area_6 = get_s2_cell_area(level=6)
        area_8 = get_s2_cell_area(level=8)
        
        # Higher level = smaller cells
        assert area_4 > area_6 > area_8
    
    def test_area_reasonable_values(self):
        """Test that area values are in reasonable ranges."""
        for level in range(4, 9):
            area = get_s2_cell_area(level=level)
            
            # Should be positive
            assert area > 0
            
            # Should not be absurdly large or small
            assert 0.01 < area < 10000


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_geometry_crossing_antimeridian(self):
        """Test handling of geometries crossing the antimeridian."""
        # Geometry crossing 180° longitude
        bbox = box(175, 0, -175, 10)
        
        # Should handle without error
        tokens = get_bounding_box_s2_covering_tokens(bbox, level=6)
        assert len(tokens) > 0
    
    def test_geometry_at_poles(self):
        """Test handling of polar regions."""
        # Near North Pole
        north_box = box(-180, 85, 180, 89)
        tokens_north = get_bounding_box_s2_covering_tokens(north_box, level=6)
        
        assert len(tokens_north) > 0
    
    def test_empty_geometry(self):
        """Test handling of empty geometry."""
        empty_polygon = Polygon()
        
        with pytest.raises((ValueError, AttributeError)):
            get_bounding_box_s2_covering_tokens(empty_polygon, level=6)
    
    def test_point_geometry(self):
        """Test handling of point geometry."""
        point = Point(0, 0)
        
        tokens = get_bounding_box_s2_covering_tokens(point, level=6)
        
        # Should generate at least one token for the point
        assert len(tokens) >= 1
    
    def test_very_small_geometry(self):
        """Test handling of very small geometry (building-scale)."""
        # Roughly 10m x 10m
        tiny_box = box(0.0, 0.0, 0.0001, 0.0001)
        
        tokens = get_bounding_box_s2_covering_tokens(tiny_box, level=6)
        
        # Should still generate token(s)
        assert len(tokens) >= 1


class TestS2Integration:
    """Integration tests for S2 utilities."""
    
    def test_token_generation_and_conversion_roundtrip(self):
        """Test generating tokens and converting back to polygons."""
        original_bbox = box(-5, 40, 5, 50)
        
        # Generate tokens
        tokens = get_bounding_box_s2_covering_tokens(original_bbox, level=6)
        
        # Convert tokens back to polygons
        polygons = [s2_token_to_shapely_polygon(t) for t in tokens]
        
        # Verify all conversions succeeded
        assert len(polygons) == len(tokens)
        assert all(p.is_valid for p in polygons)
        
        # Union should cover a significant portion of original
        coverage = unary_union(polygons)
        intersection = coverage.intersection(original_bbox)
        
        # Coverage should be substantial
        assert intersection.area > 0
    
    def test_consistent_token_generation(self):
        """Test that token generation is deterministic."""
        bbox = box(0, 0, 1, 1)
        
        tokens1 = get_bounding_box_s2_covering_tokens(bbox, level=6)
        tokens2 = get_bounding_box_s2_covering_tokens(bbox, level=6)
        
        # Should generate same tokens for same input
        assert set(tokens1) == set(tokens2)
    
    def test_realistic_city_coverage(self):
        """Test realistic scenario: covering a city-sized area."""
        # Roughly New York City size (~1000 km²)
        city_box = box(-74.05, 40.68, -73.90, 40.88)
        
        tokens = get_bounding_box_s2_covering_tokens(city_box, level=6)
        
        # Should generate reasonable number of tokens
        assert 1 < len(tokens) < 100
        
        # Estimate should be reasonable
        estimate = estimate_cells_for_geometry(city_box, level=6)
        assert 1 < estimate < 100