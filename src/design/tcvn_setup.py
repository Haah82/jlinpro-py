"""
TCVN 5574:2018 Material Properties and Design Context
Vietnamese concrete design standard material definitions and setup
"""

from typing import Tuple, List, Optional
from pydantic import BaseModel, Field, field_validator
import numpy as np


class ConcreteGrade(BaseModel):
    """
    TCVN 5574:2018 Concrete Grade Properties.
    
    Reference: TCVN 5574:2018 Section 6 - Material Properties
    
    Attributes:
        name: Grade designation (e.g., "B20", "B30")
        Rb: Design compressive strength (MPa)
        Rbt: Design tensile strength (MPa)
        Eb: Modulus of elasticity (MPa)
        gamma_b: Partial safety factor for concrete (default 1.0)
    
    Example:
        >>> concrete = ConcreteGrade(name="B25", Rb=14.5, Rbt=1.05, Eb=30000)
        >>> print(f"Concrete {concrete.name}: Rb = {concrete.Rb} MPa")
        Concrete B25: Rb = 14.5 MPa
    """
    name: str = Field(..., description="Concrete grade designation")
    Rb: float = Field(..., gt=0, description="Design compressive strength (MPa)")
    Rbt: float = Field(..., gt=0, description="Design tensile strength (MPa)")
    Eb: float = Field(..., gt=0, description="Modulus of elasticity (MPa)")
    gamma_b: float = Field(default=1.0, gt=0, description="Partial safety factor")
    
    @field_validator('Rb')
    @classmethod
    def validate_rb(cls, v: float) -> float:
        """Validate Rb is within typical range."""
        if not (5.0 <= v <= 40.0):
            raise ValueError(f"Rb = {v} MPa is outside typical range [5, 40] MPa")
        return v
    
    @field_validator('Rbt')
    @classmethod
    def validate_rbt(cls, v: float) -> float:
        """Validate Rbt is positive and reasonable."""
        if v > 5.0:
            raise ValueError(f"Rbt = {v} MPa seems too high (typical < 5 MPa)")
        return v
    
    @property
    def fc(self) -> float:
        """
        Cylinder compressive strength (approximate).
        For conversion to other codes that use fc'.
        """
        return self.Rb * 1.25  # Approximate conversion
    
    def get_stress_strain_model(self, model_type: str = 'bilinear') -> Tuple[List[float], List[float]]:
        """
        Get stress-strain relationship for concrete.
        
        Args:
            model_type: 'bilinear', 'parabolic', or 'rectangular'
        
        Returns:
            (strains, stresses): Lists of strain and stress values
        
        Example:
            >>> concrete = CONCRETE_GRADES['B25']
            >>> strains, stresses = concrete.get_stress_strain_model('bilinear')
            >>> print(f"Peak stress: {max(stresses):.1f} MPa")
        """
        if model_type == 'bilinear':
            # Simplified bilinear model
            epsilon_c0 = 0.002  # Strain at peak stress
            epsilon_cu = 0.0035  # Ultimate strain
            
            strains = [0.0, epsilon_c0, epsilon_cu]
            stresses = [0.0, self.Rb, 0.85 * self.Rb]
            
        elif model_type == 'parabolic':
            # Parabolic stress block (Hognestad)
            epsilon_c0 = 0.002
            epsilon_cu = 0.0035
            
            strains = np.linspace(0, epsilon_cu, 50)
            stresses = []
            
            for eps in strains:
                if eps <= epsilon_c0:
                    # Ascending branch (parabola)
                    sigma = self.Rb * (2 * eps / epsilon_c0 - (eps / epsilon_c0)**2)
                else:
                    # Descending branch (linear)
                    sigma = self.Rb * (1 - 0.15 * (eps - epsilon_c0) / (epsilon_cu - epsilon_c0))
                stresses.append(max(sigma, 0))
            
            strains = strains.tolist()
            
        elif model_type == 'rectangular':
            # Whitney rectangular stress block
            epsilon_cu = 0.0035
            strains = [0.0, epsilon_cu]
            stresses = [0.85 * self.Rb, 0.85 * self.Rb]
            
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        return strains, stresses
    
    class Config:
        """Pydantic configuration."""
        frozen = False


class SteelGrade(BaseModel):
    """
    TCVN 5574:2018 Reinforcing Steel Grade Properties.
    
    Reference: TCVN 5574:2018 Section 6 - Material Properties
    
    Attributes:
        name: Grade designation (e.g., "CB400V", "CB500V")
        Rs: Design tensile strength (MPa)
        Rsc: Design compressive strength (MPa)
        Es: Modulus of elasticity (MPa)
        gamma_s: Partial safety factor for steel (default 1.15)
    
    Example:
        >>> steel = SteelGrade(name="CB400V", Rs=350, Rsc=350, Es=200000)
        >>> print(f"Steel {steel.name}: fy = {steel.fy} MPa")
        Steel CB400V: fy = 350 MPa
    """
    name: str = Field(..., description="Steel grade designation")
    Rs: float = Field(..., gt=0, description="Design tensile strength (MPa)")
    Rsc: float = Field(..., gt=0, description="Design compressive strength (MPa)")
    Es: float = Field(default=200000, gt=0, description="Modulus of elasticity (MPa)")
    gamma_s: float = Field(default=1.15, gt=0, description="Partial safety factor")
    
    @field_validator('Rs')
    @classmethod
    def validate_rs(cls, v: float) -> float:
        """Validate Rs is within typical range."""
        if not (200.0 <= v <= 500.0):
            raise ValueError(f"Rs = {v} MPa is outside typical range [200, 500] MPa")
        return v
    
    @property
    def fy(self) -> float:
        """
        Yield strength (alias for Rs).
        For compatibility with other code conventions.
        """
        return self.Rs
    
    @property
    def epsilon_y(self) -> float:
        """Yield strain."""
        return self.Rs / self.Es
    
    class Config:
        """Pydantic configuration."""
        frozen = False


# Predefined concrete grades per TCVN 5574:2018 Table 6.7
CONCRETE_GRADES = {
    "B15": ConcreteGrade(name="B15", Rb=8.5, Rbt=0.75, Eb=23000),
    "B20": ConcreteGrade(name="B20", Rb=11.5, Rbt=0.90, Eb=27500),
    "B25": ConcreteGrade(name="B25", Rb=14.5, Rbt=1.05, Eb=30000),
    "B30": ConcreteGrade(name="B30", Rb=17.0, Rbt=1.20, Eb=32500),
    "B35": ConcreteGrade(name="B35", Rb=19.5, Rbt=1.30, Eb=34500),
    "B40": ConcreteGrade(name="B40", Rb=22.0, Rbt=1.40, Eb=36000),
    "B45": ConcreteGrade(name="B45", Rb=25.0, Rbt=1.50, Eb=37000),
    "B50": ConcreteGrade(name="B50", Rb=27.5, Rbt=1.60, Eb=38000),
    "B60": ConcreteGrade(name="B60", Rb=33.0, Rbt=1.80, Eb=39500)
}

# Predefined steel grades per TCVN 5574:2018 Table 6.8
STEEL_GRADES = {
    "CB240T": SteelGrade(name="CB240T", Rs=210, Rsc=210, Es=200000),
    "CB300V": SteelGrade(name="CB300V", Rs=260, Rsc=260, Es=200000),
    "CB400V": SteelGrade(name="CB400V", Rs=350, Rsc=350, Es=200000),
    "CB500V": SteelGrade(name="CB500V", Rs=435, Rsc=400, Es=200000)
}


class MaterialLoader:
    """
    Helper class to load standard TCVN 5574:2018 material properties.
    
    Provides static methods to retrieve predefined concrete and steel grades.
    
    Example:
        >>> concrete = MaterialLoader.get_concrete("B25")
        >>> steel = MaterialLoader.get_steel("CB400V")
        >>> print(f"Using {concrete.name} concrete with {steel.name} steel")
        Using B25 concrete with CB400V steel
    """
    
    @staticmethod
    def get_concrete(grade_name: str) -> ConcreteGrade:
        """
        Get predefined concrete grade.
        
        Args:
            grade_name: Grade designation (e.g., "B20", "B30")
        
        Returns:
            ConcreteGrade object
        
        Raises:
            ValueError: If grade_name not found
        """
        if grade_name not in CONCRETE_GRADES:
            available = ", ".join(CONCRETE_GRADES.keys())
            raise ValueError(
                f"Unknown concrete grade: {grade_name}. "
                f"Available grades: {available}"
            )
        return CONCRETE_GRADES[grade_name]
    
    @staticmethod
    def get_steel(grade_name: str) -> SteelGrade:
        """
        Get predefined steel grade.
        
        Args:
            grade_name: Grade designation (e.g., "CB400V", "CB500V")
        
        Returns:
            SteelGrade object
        
        Raises:
            ValueError: If grade_name not found
        """
        if grade_name not in STEEL_GRADES:
            available = ", ".join(STEEL_GRADES.keys())
            raise ValueError(
                f"Unknown steel grade: {grade_name}. "
                f"Available grades: {available}"
            )
        return STEEL_GRADES[grade_name]
    
    @staticmethod
    def list_concrete_grades() -> List[str]:
        """Get list of available concrete grade names."""
        return list(CONCRETE_GRADES.keys())
    
    @staticmethod
    def list_steel_grades() -> List[str]:
        """Get list of available steel grade names."""
        return list(STEEL_GRADES.keys())


class DesignContext(BaseModel):
    """
    Design context holding material properties and global parameters.
    
    Encapsulates all design parameters needed for TCVN 5574:2018 checks.
    
    Attributes:
        concrete: ConcreteGrade object
        steel: SteelGrade object
        cover: Concrete cover (mm)
        gamma_b: Concrete partial safety factor (override)
        gamma_s: Steel partial safety factor (override)
        environment: Exposure environment ('normal' or 'aggressive')
    
    Example:
        >>> context = DesignContext(
        ...     concrete=MaterialLoader.get_concrete("B25"),
        ...     steel=MaterialLoader.get_steel("CB400V"),
        ...     cover=25.0
        ... )
        >>> print(f"Effective cover: {context.cover} mm")
        Effective cover: 25.0 mm
    """
    concrete: ConcreteGrade = Field(..., description="Concrete grade")
    steel: SteelGrade = Field(..., description="Steel grade")
    cover: float = Field(..., gt=15.0, description="Concrete cover (mm)")
    gamma_b: float = Field(default=1.0, gt=0, description="Concrete safety factor")
    gamma_s: float = Field(default=1.15, gt=0, description="Steel safety factor")
    environment: str = Field(default='normal', description="Exposure environment")
    
    @field_validator('cover')
    @classmethod
    def validate_cover(cls, v: float) -> float:
        """Validate cover meets minimum requirements."""
        if v < 15.0:
            raise ValueError(f"Cover = {v} mm is below minimum 15 mm")
        if v > 100.0:
            raise ValueError(f"Cover = {v} mm seems excessive (typical < 100 mm)")
        return v
    
    @field_validator('environment')
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment type."""
        allowed = ['normal', 'aggressive', 'marine', 'mild']
        if v not in allowed:
            raise ValueError(f"Environment '{v}' not in allowed values: {allowed}")
        return v
    
    @classmethod
    def from_ui_inputs(
        cls,
        concrete_name: str,
        steel_name: str,
        cover: float,
        environment: str = 'normal'
    ) -> 'DesignContext':
        """
        Factory method to create context from UI selections.
        
        Args:
            concrete_name: Concrete grade name (e.g., "B25")
            steel_name: Steel grade name (e.g., "CB400V")
            cover: Concrete cover (mm)
            environment: Exposure environment
        
        Returns:
            DesignContext instance
        
        Example:
            >>> ctx = DesignContext.from_ui_inputs("B30", "CB400V", 30.0)
            >>> print(f"Context: {ctx.concrete.name} + {ctx.steel.name}")
            Context: B30 + CB400V
        """
        return cls(
            concrete=MaterialLoader.get_concrete(concrete_name),
            steel=MaterialLoader.get_steel(steel_name),
            cover=cover,
            environment=environment
        )
    
    def get_xi_limit(self) -> float:
        """
        Get relative compression zone depth limit ξ_R.
        
        Per TCVN 5574:2018 Table 8.1, depends on steel grade:
        - CB240T, CB300V: ξ_R = 0.60
        - CB400V: ξ_R = 0.55
        - CB500V: ξ_R = 0.50
        
        Returns:
            ξ_R limit value
        """
        steel_name = self.steel.name
        
        if steel_name == "CB240T" or steel_name == "CB300V":
            return 0.60
        elif steel_name == "CB400V":
            return 0.55
        elif steel_name == "CB500V":
            return 0.50
        else:
            # Conservative default
            return 0.50
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True


def create_design_context_from_streamlit(
    concrete_name: str,
    steel_name: str,
    cover: float,
    environment: str = 'normal'
) -> DesignContext:
    """
    Convenience function for Streamlit integration.
    
    Called from app/pages/design.py to create DesignContext from UI selections.
    
    Args:
        concrete_name: Selected concrete grade (e.g., "B25")
        steel_name: Selected steel grade (e.g., "CB400V")
        cover: Concrete cover in mm
        environment: Exposure environment
    
    Returns:
        DesignContext instance ready for design checks
    
    Raises:
        ValueError: If invalid grade names or parameters
    
    Example:
        >>> # In Streamlit app
        >>> concrete_grade = st.selectbox("Concrete", ["B15", "B20", "B25"])
        >>> steel_grade = st.selectbox("Steel", ["CB240T", "CB400V"])
        >>> cover = st.number_input("Cover (mm)", value=25.0)
        >>> 
        >>> context = create_design_context_from_streamlit(
        ...     concrete_grade, steel_grade, cover
        ... )
    """
    return DesignContext(
        concrete=MaterialLoader.get_concrete(concrete_name),
        steel=MaterialLoader.get_steel(steel_name),
        cover=cover,
        gamma_b=1.0,
        gamma_s=1.15,
        environment=environment
    )
