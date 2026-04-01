"""
Comprehensive SQS (Special Quasirandom Structures) generator module.
Uses sqsgenerator as the primary method with ATAT as fallback.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path

from pymatgen.core import Structure, Element, Composition
from ase import Atoms

logger = logging.getLogger(__name__)


class SQSGenerationError(Exception):
    """Custom exception for SQS generation errors."""
    pass


class SQSGenerator:
    """
    Unified SQS generator that uses sqsgenerator as primary method with ATAT fallback.
    """
    
    def __init__(self, 
                 method: str = 'sqsgenerator',
                 fallback_method: str = 'atat',
                 **kwargs):
        """
        Initialize the SQS generator.
        
        Args:
            method: Primary SQS generation method ('sqsgenerator' or 'atat')
            fallback_method: Fallback method if primary fails
            **kwargs: Additional configuration options
        """
        self.method = method
        self.fallback_method = fallback_method
        self.config = kwargs
        
        # Try to import sqsgenerator
        self._sqsgenerator_available = self._check_sqsgenerator_availability()
        
        # Initialize ATAT fallback
        try:
            from .mocks.atat import MockATATInterface
        except ImportError:
            from mocks.atat import MockATATInterface
        self._atat_interface = MockATATInterface()
        
        logger.info(f"Initialized SQS generator with method: {method}")
        logger.info(f"sqsgenerator available: {self._sqsgenerator_available}")
    
    def _check_sqsgenerator_availability(self) -> bool:
        """Check if sqsgenerator is available."""
        try:
            import sqsgenerator
            logger.info(f"sqsgenerator version available")
            return True
        except ImportError as e:
            logger.warning(f"sqsgenerator not available: {e}")
            return False
    
    def generate_sqs(self,
                    base_structure: Structure,
                    alloy_spec: str,
                    composition: float,
                    supercell: Tuple[int, int, int] = (2, 2, 2),
                    num_structures: int = 3,
                    **kwargs) -> List[Structure]:
        """
        Generate SQS structures using the configured method.
        
        Args:
            base_structure: Base crystal structure
            alloy_spec: Alloy specification (e.g., "K:Na")
            composition: Target composition (0.0 to 1.0)
            supercell: Supercell dimensions
            num_structures: Number of SQS structures to generate
            **kwargs: Additional method-specific parameters
            
        Returns:
            List of SQS structures
            
        Raises:
            SQSGenerationError: If both primary and fallback methods fail
        """
        logger.info(f"Generating {num_structures} SQS structures for {alloy_spec}, x={composition}")
        
        # Try primary method
        try:
            if self.method == 'sqsgenerator' and self._sqsgenerator_available:
                return self._generate_with_sqsgenerator(
                    base_structure, alloy_spec, composition, supercell, num_structures, **kwargs
                )
            elif self.method == 'atat':
                return self._generate_with_atat(
                    base_structure, alloy_spec, composition, supercell, num_structures, **kwargs
                )
            else:
                raise SQSGenerationError(f"Primary method '{self.method}' not available")
                
        except Exception as e:
            logger.warning(f"Primary method '{self.method}' failed: {e}")
            
            # Try fallback method
            try:
                logger.info(f"Attempting fallback to '{self.fallback_method}'")
                if self.fallback_method == 'atat':
                    return self._generate_with_atat(
                        base_structure, alloy_spec, composition, supercell, num_structures, **kwargs
                    )
                elif self.fallback_method == 'sqsgenerator' and self._sqsgenerator_available:
                    return self._generate_with_sqsgenerator(
                        base_structure, alloy_spec, composition, supercell, num_structures, **kwargs
                    )
                else:
                    raise SQSGenerationError(f"Fallback method '{self.fallback_method}' not available")
                    
            except Exception as fallback_error:
                logger.error(f"Fallback method '{self.fallback_method}' also failed: {fallback_error}")
                raise SQSGenerationError(
                    f"Both primary method '{self.method}' and fallback '{self.fallback_method}' failed. "
                    f"Primary error: {e}. Fallback error: {fallback_error}"
                )
    
    def _generate_with_sqsgenerator(self,
                                   base_structure: Structure,
                                   alloy_spec: str,
                                   composition: float,
                                   supercell: Tuple[int, int, int],
                                   num_structures: int,
                                   **kwargs) -> List[Structure]:
        """Generate SQS using sqsgenerator (v0.4+ API)."""
        try:
            from sqsgenerator import optimize, from_pymatgen, to_pymatgen
        except ImportError:
            raise SQSGenerationError("sqsgenerator not available")
        
        logger.info("Using sqsgenerator for SQS generation")
        
        # Parse alloy specification
        original_element, substitute_element = alloy_spec.split(":")
        
        # Create supercell
        supercell_structure = base_structure.copy()
        supercell_structure.make_supercell(supercell)
        
        # Convert to sqsgenerator format
        sqs_structure = from_pymatgen(supercell_structure)
        
        # Find sites with the original element for composition calculation
        target_sites = []
        for i, site in enumerate(supercell_structure.sites):
            if Element(original_element) in site.species:
                target_sites.append(i)
        
        if not target_sites:
            raise SQSGenerationError(f"No sites found with element {original_element}")
        
        # Calculate composition dictionary
        num_to_substitute = int(len(target_sites) * composition)
        num_original = len(target_sites) - num_to_substitute
        
        composition_dict = {
            substitute_element: num_to_substitute,
            original_element: num_original
        }
        
        logger.info(f"Target composition: {composition_dict}")
        
        # For the new API, let's try a simpler approach using YAML-like config
        config_dict = {
            'structure': {
                'lattice': supercell_structure.lattice.matrix.tolist(),
                'coords': [site.frac_coords.tolist() for site in supercell_structure.sites],
                'species': [str(site.specie) for site in supercell_structure.sites]
            },
            'composition': composition_dict,
            'which': original_element,
            'iterations': int(kwargs.get('iterations', 1e7)),
            'shell_weights': kwargs.get('shell_weights', {1: 1.0}),
            'max_output_configurations': max(num_structures, 10),
            'mode': kwargs.get('mode', 'random')
        }
        
        logger.info(f"Running sqsgenerator with {config_dict['iterations']:.0e} iterations")
        
        try:
            # Parse configuration first
            from sqsgenerator import parse_config
            parsed_config = parse_config(config_dict)
            
            # Run SQS optimization using new API
            result_pack = optimize(parsed_config)
            
            logger.info(f"sqsgenerator completed, found {len(result_pack.results)} configurations")
            
            # Convert results to pymatgen structures
            sqs_structures = []
            
            # Sort results by objective value and take the best ones
            sorted_results = sorted(result_pack.results.items(), 
                                  key=lambda x: x[1].objective if hasattr(x[1], 'objective') else 0)
            
            for i, (rank, result) in enumerate(sorted_results[:num_structures]):
                try:
                    # Convert structure back to pymatgen
                    if hasattr(result, 'structure'):
                        structure = to_pymatgen(result.structure)
                    else:
                        # Fallback: reconstruct from configuration if available
                        structure = self._reconstruct_structure_from_config(
                            supercell_structure, getattr(result, 'configuration', []), 
                            original_element, substitute_element
                        )
                    
                    sqs_structures.append(structure)
                    
                    objective = getattr(result, 'objective', 'N/A')
                    logger.info(f"SQS {i+1}: rank={rank}, objective={objective}")
                    
                except Exception as e:
                    logger.warning(f"Failed to convert result {i+1}: {e}")
                    continue
            
            if not sqs_structures:
                raise SQSGenerationError("No valid structures could be generated")
            
            return sqs_structures
            
        except Exception as e:
            logger.error(f"sqsgenerator optimization failed: {e}")
            raise SQSGenerationError(f"sqsgenerator optimization failed: {e}")
    
    def _generate_with_atat(self,
                           base_structure: Structure,
                           alloy_spec: str,
                           composition: float,
                           supercell: Tuple[int, int, int],
                           num_structures: int,
                           **kwargs) -> List[Structure]:
        """Generate SQS using ATAT (mock implementation)."""
        logger.info("Using ATAT (mock) for SQS generation")
        
        return self._atat_interface.generate_sqs(
            base_structure=base_structure,
            alloy_spec=alloy_spec,
            composition=composition,
            supercell=supercell,
            num_structures=num_structures
        )
    
    def _reconstruct_structure_from_config(self,
                                         base_structure: Structure,
                                         configuration: List[str],
                                         original_element: str,
                                         substitute_element: str) -> Structure:
        """Reconstruct pymatgen Structure from sqsgenerator configuration."""
        structure = base_structure.copy()
        
        # Find sites with the original element
        target_sites = []
        for i, site in enumerate(structure.sites):
            if Element(original_element) in site.species:
                target_sites.append(i)
        
        # Apply configuration
        for i, target_site_idx in enumerate(target_sites):
            if i < len(configuration):
                new_element = configuration[i]
                if new_element != original_element:
                    structure.replace(target_site_idx, Element(new_element))
        
        return structure
    
    def analyze_sqs_quality(self,
                           structures: List[Structure],
                           shell_weights: Optional[Dict[int, float]] = None) -> Dict[str, Any]:
        """
        Analyze the quality of generated SQS structures.
        
        Args:
            structures: List of SQS structures to analyze
            shell_weights: Shell weights for SRO parameter calculation
            
        Returns:
            Dictionary containing quality metrics
        """
        if not self._sqsgenerator_available:
            logger.warning("sqsgenerator not available for SQS quality analysis")
            return {"error": "sqsgenerator not available"}
        
        try:
            # For now, return basic metrics since the analysis API has changed
            # This is a simplified implementation for the new API
            quality_metrics = {
                'num_structures': len(structures),
                'compositions': [struct.composition.reduced_formula for struct in structures],
                'num_sites': [struct.num_sites for struct in structures],
                'note': 'Full SRO analysis requires updated API implementation'
            }
            
            logger.info(f"Basic SQS quality analysis completed for {len(structures)} structures")
            return quality_metrics
            
        except Exception as e:
            logger.error(f"SQS quality analysis failed: {e}")
            return {"error": str(e)}
    
    def get_method_info(self) -> Dict[str, Any]:
        """Get information about available methods and current configuration."""
        return {
            'primary_method': self.method,
            'fallback_method': self.fallback_method,
            'sqsgenerator_available': self._sqsgenerator_available,
            'atat_available': True,  # Mock is always available
            'configuration': self.config
        }


# Convenience functions for backward compatibility
def generate_sqs_structures(base_structure: Structure,
                          alloy_spec: str,
                          composition: float,
                          supercell: Tuple[int, int, int] = (2, 2, 2),
                          num_structures: int = 3,
                          method: str = 'sqsgenerator',
                          **kwargs) -> List[Structure]:
    """
    Convenience function to generate SQS structures.
    
    Args:
        base_structure: Base crystal structure
        alloy_spec: Alloy specification (e.g., "K:Na")
        composition: Target composition (0.0 to 1.0)
        supercell: Supercell dimensions
        num_structures: Number of SQS structures to generate
        method: SQS generation method ('sqsgenerator' or 'atat')
        **kwargs: Additional method-specific parameters
        
    Returns:
        List of SQS structures
    """
    generator = SQSGenerator(method=method, **kwargs)
    return generator.generate_sqs(
        base_structure, alloy_spec, composition, supercell, num_structures, **kwargs
    )

