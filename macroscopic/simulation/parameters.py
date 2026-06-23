import casadi as cs
import numpy as np

class BaseParameters:
    '''
    Base class for parameters used in macroscopic traffic flow models.
    This class provides a common structure for defining parameters and allows for easy extension by subclasses.
    '''
    def __new__(cls, *args, **kwargs):
        # Merge DEFAULTS from all parent classes
        merged_defaults = {}
        for base in reversed(cls.__mro__):
            if hasattr(base, 'DEFAULTS') and base is not object:
                merged_defaults.update(base.DEFAULTS)
        cls.DEFAULTS = merged_defaults
        
        # Merge __slots__ from all parent classes
        merged_slots = []
        for base in reversed(cls.__mro__):
            if hasattr(base, '__slots__') and base is not object:
                merged_slots.extend(base.__slots__)
        # Remove duplicates while preserving order
        cls.__slots__ = tuple(dict.fromkeys(merged_slots))
        
        return super().__new__(cls)
    
    # defintions from: https://web.dcsc.tudelft.nl/~bdeschutter/research/phd_theses/phd_hegyi_2004.pdf
    DEFAULTS = {
        "T": 10 / 3600, # time step size of the freeway simulation, typical value is 10 seconds
        "Tfin": 2.5, # changed down from 2.5 to 1
        "L": 1,  # segment length in km (changed from 1000 to match main.py)
        "lanes": 2,
        # "C": (3800, 1800), # MAX MAINLINE: 3700, MAX RAMP: 1800
        "C": (3800, 1800), # veh/h, maximum flow capacity per lane for mainline and on-ramp
        "tau": 18 / 3600, # 18/3600 # time constant in hours
        "kappa": 40, # veh/km/lane
        "eta": 60, # km^2/hour
        "rho_max": 180, # veh/km/lane
        "delta": 0.0122, 
        "rho_crit": 33.5, # veh/km/lane
        "v_free": 112, # km/h
        "a": 2.5,
        # initial conditions
        "rho_init": None,
        "v_free_init": None,
        "w_init": None,
        "r": None,
    }
    
    __slots__ = (
        "_T",
        "_Tfin",
        "_time",
        "_L",
        "_lanes",
        "_C",
        "_tau",
        "_kappa",
        "_eta",
        "_rho_max",
        "_delta",
        "_rho_crit",
        "_a",
        
        "_v_free",
        "_rho_init",
        "_v_free_init",
        "_w_init",
        "_r_init",
        
        "_dynamic_params",
        "_param_name",
    )
    
    def __init__(self, T=None, Tfin=None, time=None, L=None, lanes=None,
                 C=None, tau=None, kappa=None, eta=None, rho_max=None,
                 delta=None, rho_crit=None, v_free=None, a=None,
                 
                 rho_init=None, v_free_init=None, w_init=None, r_init=None
                 ):
        self._T = BaseParameters.DEFAULTS["T"] if T is None else T
        self._Tfin = BaseParameters.DEFAULTS["Tfin"] if Tfin is None else Tfin
        self._time = np.arange(0, self._Tfin, self._T) if time is None else time
        self._L = BaseParameters.DEFAULTS["L"] if L is None else L
        self._lanes = BaseParameters.DEFAULTS["lanes"] if lanes is None else lanes
        self._C = BaseParameters.DEFAULTS["C"] if C is None else C
        self._tau = BaseParameters.DEFAULTS["tau"] if tau is None else tau
        self._kappa = BaseParameters.DEFAULTS["kappa"] if kappa is None else kappa
        self._eta = BaseParameters.DEFAULTS["eta"] if eta is None else eta
        self._rho_max = BaseParameters.DEFAULTS["rho_max"] if rho_max is None else rho_max
        self._delta = BaseParameters.DEFAULTS["delta"] if delta is None else delta
        self._rho_crit = BaseParameters.DEFAULTS["rho_crit"] if rho_crit is None else rho_crit
        self._v_free = BaseParameters.DEFAULTS["v_free"] if v_free is None else v_free
        self._a = BaseParameters.DEFAULTS.get("a", None) if a is None else a
        # initial conditions
        self._rho_init = rho_init
        self._v_free_init = v_free_init
        self._w_init = w_init
        self._r_init = r_init
        
        self._param_name = ""
        
        assert rho_init is not None, "Initial density (rho_init) must be provided"
        assert v_free_init is not None, "Initial free flow speed (v_free_init) must be provided"
        assert w_init is not None, "Initial congestion queue (w_init) must be provided"
        assert r_init is not None, "Initial ramp metering rate (r_init) must be provided"

    @property
    def T(self) -> float:
        '''
        Macroscopic time step size in hours. Default is 10 seconds (10/3600 hours).
        '''
        return self._T
    
    @T.setter
    def T(self, value: float):
        if value <= 0:
            raise ValueError("Time step size (T) must be positive.")
        self._T = value

    @property
    def Tfin(self) -> float:
        '''
        Final time in hours. Must be positive.
        '''
        return self._Tfin
    
    @Tfin.setter
    def Tfin(self, value: float):
        if value <= 0:
            raise ValueError("Final time (Tfin) must be positive.")
        self._Tfin = value

    @property
    def time(self):
        '''
        Time array from 0 to Tfin with step size T.
        '''
        return self._time
    
    @time.setter
    def time(self, value: np.ndarray):
        if not isinstance(value, np.ndarray):
            raise ValueError("Time must be a numpy array.")
        self._time = value

    @property
    def L(self) -> float:
        '''
        Segment length in kilometers. Must be positive.
        '''
        return self._L
    
    @L.setter
    def L(self, value: float):
        if value <= 0:
            raise ValueError("Segment length (L) must be positive.")
        self._L = value

    @property
    def lanes(self) -> int:
        '''
        Number of lanes. Must be a positive integer.
        '''
        return self._lanes
    
    @lanes.setter
    def lanes(self, value: int):
        if not isinstance(value, int) or value <= 0:
            raise ValueError("Number of lanes must be a positive integer.")
        self._lanes = value

    @property
    def C(self) -> tuple:   
        ''' 
        Maximum flow capacity per lane for mainline and on-ramp in vehicles per hour.
        Must be a tuple of two positive numbers.
        '''
        return self._C
    
    @C.setter
    def C(self, value: tuple):
        if not isinstance(value, tuple) or len(value) != 2 or any(v <= 0 for v in value):
            raise ValueError("C must be a tuple of two positive numbers (mainline, on-ramp).")
        self._C = value

    @property
    def tau(self) -> float: 
        '''
        Time constant in hours. Must be positive.
        '''
        return self._tau
    
    @tau.setter
    def tau(self, value: float):
        if value <= 0:
            raise ValueError("Time constant (tau) must be positive.")
        self._tau = value

    @property
    def kappa(self) -> float:
        '''
        Parameter kappa in vehicles per kilometer per lane. Must be positive.
        '''
        return self._kappa  
    
    @kappa.setter
    def kappa(self, value: float):
        if value <= 0:
            raise ValueError("Parameter kappa must be positive.")
        self._kappa = value

    @property
    def eta(self) -> float:
        '''
        Parameter eta in km^2/hour. Must be positive.
        '''
        return self._eta
    
    @eta.setter
    def eta(self, value: float):    
        
        if value <= 0:
            raise ValueError("Parameter eta must be positive.")
        self._eta = value

    @property
    def rho_max(self) -> float:
        '''
        Maximum density in vehicles per kilometer per lane. Must be positive.
        '''
        return self._rho_max
    
    @rho_max.setter
    def rho_max(self, value: float):
        if value <= 0:
            raise ValueError("Maximum density (rho_max) must be positive.")
        self._rho_max = value

    @property
    def delta(self) -> float:
        '''
        Parameter delta. Must be positive.
        '''
        return self._delta
    
    @delta.setter
    def delta(self, value: float):
        if value <= 0:
            raise ValueError("Parameter delta must be positive.")
        self._delta = value

    @property
    def rho_crit(self) -> float:
        '''
        Critical density in vehicles per kilometer per lane. Must be positive.
        '''
        return self._rho_crit
    
    @rho_crit.setter
    def rho_crit(self, value: float):
        if value <= 0:
            raise ValueError("Critical density (rho_crit) must be positive.")
        self._rho_crit = value

    @property
    def v_free(self) -> float:
        '''
        Free flow speed in kilometers per hour. Must be positive.
        '''
        return self._v_free
    
    @v_free.setter
    def v_free(self, value: float):
        if value <= 0:
            raise ValueError("Free flow speed (v_free) must be positive.")
        self._v_free = value

    @property
    def a(self) -> float:
        '''
        Parameter a. Must be positive.
        '''
        return self._a
    
    @a.setter
    def a(self, value: float):
        if value <= 0:
            raise ValueError("Parameter a must be positive.")
        self._a = value 

    @property
    def rho_init(self):
        '''
        Initial density in vehicles per kilometer per lane. Must be provided.
        '''
        return self._rho_init
    
    @rho_init.setter
    def rho_init(self, value):
        if value is None:
            raise ValueError("Initial density (rho_init) must be provided.")
        if not hasattr(value, '__iter__'):
            raise ValueError("Initial density (rho_init) must be an iterable (e.g., list or numpy array).")
        
        self._rho_init = value

    @property
    def v_free_init(self):
        '''
        Initial free flow speed in kilometers per hour. Must be provided.
        '''
        return self._v_free_init
    
    @v_free_init.setter
    def v_free_init(self, value):
        if value is None:
            raise ValueError("Initial free flow speed (v_free_init) must be provided.")
        if not hasattr(value, '__iter__'):
            raise ValueError("Initial free flow speed (v_free_init) must be an iterable (e.g., list or numpy array).")
        self._v_free_init = value

    @property
    def w_init(self):
        '''
        Initial congestion queue. Must be provided.
        '''
        return self._w_init
    
    @w_init.setter
    def w_init(self, value):
        if value is None:
            raise ValueError("Initial congestion queue (w_init) must be provided.")
        if not hasattr(value, '__iter__'):
            raise ValueError("Initial congestion queue (w_init) must be an iterable (e.g., list or numpy array).")
        self._w_init = value

    @property
    def r_init(self):
        '''
        Initial ramp metering rate. Must be provided.
        '''
        return self._r_init
    
    @r_init.setter
    def r_init(self, value):
        if value is None:
            raise ValueError("Initial ramp metering rate (r_init) must be provided.")
        if not hasattr(value, '__iter__'):
            raise ValueError("Initial ramp metering rate (r_init) must be an iterable (e.g., list or numpy array).")
        self._r_init = value 
    
    # helper methods

    def __getitem__(self, key):
        if key in self.__slots__:
            return getattr(self, key)
        raise KeyError(f"{key} not found in Parameters")

    def __setitem__(self, key, value):
        if key in self.__slots__:
            setattr(self, key, value)
        else:
            raise KeyError(f"{key} not found in Parameters") 
               
    def to_dict(self, include=None) -> dict:
        '''
        Convert the parameters to a dictionary. If 'include' is provided, only include those attributes.
        
        Args:
            include (list, optional): List of attribute names to include in the dictionary.
        Returns:
            dict: Dictionary representation of the parameters.
        '''
        if include is None:
            include = self.__slots__
        return {slot: getattr(self, slot) for slot in include if slot.startswith('_') is False}
    
    def mark_dynamic(self, **params):
        """
        Mark parameters as dynamic for sensitivity testing.
        Returns self for chaining.
        
        Example:
            params.mark_dynamic(a=True, A=True, E=True)

        Args:
            **params: Keyword arguments where keys are parameter names and values are booleans indicating if
            the parameter is dynamic.
        Returns:
            self: The instance of the class, allowing for method chaining.
        """
        if not hasattr(self, '_dynamic_params'):
            self._dynamic_params = set()
        
        for param_name, is_dynamic in params.items():
            if is_dynamic and hasattr(self, param_name):
                self._dynamic_params.add(param_name)
            elif not is_dynamic and param_name in getattr(self, '_dynamic_params', set()):
                self._dynamic_params.discard(param_name)
        
        return self
    
    def is_dynamic(self, param_name: str) -> bool:
        """Check if a parameter is marked as dynamic."""
        return param_name in getattr(self, '_dynamic_params', set())
    
    def get_dynamic_params(self) -> dict:
        """Get all dynamic parameters and their current values."""
        dynamic_params = {}
        for param in getattr(self, '_dynamic_params', set()):
            if hasattr(self, param):
                dynamic_params[param] = getattr(self, param)
        return dynamic_params
    
    def set_param_name(self, _param_name: str):
        self._param_name = _param_name
        
    def pprint(self, ignore=None):
        '''
        Pretty print the parameters, optionally ignoring specified attributes.
        
        Args:
            ignore (set, optional): Set of attribute names to ignore during printing.
        
        '''
        if ignore is None:
            ignore = set()

        for slot in self.__slots__:
            if not slot.startswith('_') and slot not in ignore:
                print(f"{slot}: {getattr(self, slot)}")
        
    @property
    def name(self) -> str:
        return self._param_name
        
class VslParameters(BaseParameters):
    
    DEFAULTS = {
        "_alpha_compliance": 0.5,
        "_A": 0.4,
        "_E": 1.5,
        "_V": 102,
        "_v_ctrl": cs.DM(102),
        "_segments_with_vsl": [],

        "_minimum_speed_limit": 60,
        "_maximum_speed_limit": 102,
    }
    
    __slots__ = (
        "_alpha_compliance",
        "_A",
        "_E",
        "_V",
        "_v_ctrl",
        "_segments_with_vsl",
        "_minimum_speed_limit",
        "_maximum_speed_limit",
    )
    
    def __init__(self, alpha_compliance=None, A=None, E=None, V=None, v_ctrl=None, segments_with_vsl=None, minimum_speed_limit=None, maximum_speed_limit=None, **kwargs):
        super().__init__(**kwargs)
        # Extra parameters in reference to https://linkinghub.elsevier.com/retrieve/pii/S2405896318307791 and 
        # http://ieeexplore.ieee.org/document/5873154/
        self.alpha_compliance = VslParameters.DEFAULTS["alpha_compliance"] if alpha_compliance is None else alpha_compliance
        self.A = VslParameters.DEFAULTS["A"] if A is None else A
        self.E = VslParameters.DEFAULTS["E"] if E is None else E
        self.V = VslParameters.DEFAULTS["V"] if V is None else V
        self.v_ctrl = VslParameters.DEFAULTS["v_ctrl"] if v_ctrl is None else v_ctrl
        self.segments_with_vsl = VslParameters.DEFAULTS["segments_with_vsl"] if segments_with_vsl is None else segments_with_vsl
        self.minimum_speed_limit = VslParameters.DEFAULTS["minimum_speed_limit"] if minimum_speed_limit is None else minimum_speed_limit
        self.maximum_speed_limit = VslParameters.DEFAULTS["maximum_speed_limit"] if maximum_speed_limit is None else maximum_speed_limit
        
    @property
    def alpha_compliance(self) -> float:
        '''
        Compliance factor for VSL. Must be between 0 and 1.
        '''
        return self._alpha_compliance
    
    @alpha_compliance.setter
    def alpha_compliance(self, value: float):
        if not (0 <= value <= 1):
            raise ValueError("Compliance factor (alpha_compliance) must be between 0 and 1.")
        self._alpha_compliance = value

    @property
    def A(self) -> float:
        '''
        Parameter A for VSL model. Must be positive.
        '''
        return self._A
    
    @A.setter
    def A(self, value: float):
        if value <= 0:
            raise ValueError("Parameter A must be positive.")
        self._A = value

    @property
    def E(self) -> float:
        '''
        Parameter E for VSL model. Must be positive.
        '''
        return self._E
    
    @E.setter
    def E(self, value: float):
        if value <= 0:
            raise ValueError("Parameter E must be positive.")
        self._E = value

    @property
    def V(self) -> float:
        '''
        Reference speed for VSL model. Must be positive.
        '''
        return self._V
    
    @V.setter
    def V(self, value: float):
        if value <= 0:
            raise ValueError("Reference speed (V) must be positive.")
        self._V = value

    @property
    def v_ctrl(self) -> cs.DM:
        '''
        Control speed for VSL. Must be positive.
        '''
        return self._v_ctrl
    
    @v_ctrl.setter
    def v_ctrl(self, value: cs.DM):
        if value <= 0:
            raise ValueError("Control speed (v_ctrl) must be positive.")
        self._v_ctrl = value

    @property
    def segments_with_vsl(self) -> list:
        '''
        List of segments with VSL. Must be a list of integers.
        '''
        return self._segments_with_vsl
    
    @segments_with_vsl.setter
    def segments_with_vsl(self, value: list):
        if not isinstance(value, list) or not all(isinstance(i, int) for i in value):
            raise ValueError("Segments with VSL must be a list of integers.")
        self._segments_with_vsl = value

    @property
    def minimum_speed_limit(self) -> float:
        '''
        Minimum speed limit for VSL. Must be positive. (Km/h)
        '''
        return self._minimum_speed_limit
    
    @minimum_speed_limit.setter
    def minimum_speed_limit(self, value: float):
        if value <= 0:
            raise ValueError("Minimum speed limit must be positive. (Km/h)")
        self._minimum_speed_limit = value

    @property
    def maximum_speed_limit(self) -> float:
        '''
        Maximum speed limit for VSL. Must be positive. (Km/h)
        '''
        return self._maximum_speed_limit
    
    @maximum_speed_limit.setter
    def maximum_speed_limit(self, value: float):
        if value <= 0:
            raise ValueError("Maximum speed limit must be positive. (Km/h)")
        self._maximum_speed_limit = value

    