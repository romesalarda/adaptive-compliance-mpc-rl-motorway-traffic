from sym_metanet import (
    LinkWithVsl
)

class CarlsonLinkWithVsl(LinkWithVsl):
    '''
    override the equilibrium speed calculation to implement Carlson et al. VSL model
    VIA "paramater manipulation through A and E model params"

    as referenced from http://ieeexplore.ieee.org/document/5873154/
    '''
    
    def __init__(self, *args, segments_with_vsl, alpha, A=None, E=None, V=None, **kwargs):
        super().__init__(*args, segments_with_vsl=segments_with_vsl, alpha=alpha, **kwargs)
        self.A = A
        self.E = E
        self.V = V
    
    def _get_equilibrium_speed(self, engine, rho):
        
        Veq_default = engine.links.Veq(rho, self.v_free, self.rho_crit, self.a)
        ratio = self.actions["v_ctrl"] / self.V
        v_star = ratio * self.v_free
        rho_star = self.rho_crit * (1 + self.A * (1 - ratio))
        a_star = self.a * (self.E - (self.E - 1) * ratio)
        rho_seg = rho[self.vsl]
        Veq_seg = engine.links.Veq(rho_seg, v_star, rho_star, a_star)
        Veq_new = Veq_default
        Veq_new[self.vsl] = Veq_seg
        
        return Veq_new


class FrejoLinkWithVsl(LinkWithVsl):
    '''
    override the equilibrium speed calculation to implement Frejo et al. VSL model
    VIA "paramater manipulation through A and E model params AND compliance adjustment"

    as referenced from https://linkinghub.elsevier.com/retrieve/pii/S2405896318307791
    '''

    def __init__(self, *args, segments_with_vsl, alpha, A=None, E=None, V=None, **kwargs):
        super().__init__(*args, segments_with_vsl=segments_with_vsl, alpha=alpha, **kwargs)
        self.A = A
        self.E = E
        self.V = V
            
    def _get_equilibrium_speed(self, engine, rho): 
        # Frejo et al. model referenced from equation (4)
        Veq_default = engine.links.Veq(rho, self.v_free, self.rho_crit, self.a)
        ratio = engine.min((self.actions["v_ctrl"] / self.V) * (1+self.alpha), 1.0)
        v_star = engine.min(ratio * self.V, self.v_free)
        rho_star = self.rho_crit * (1 + self.A * (1 - ratio))
        a_star = self.a * (self.E - (self.E - 1) * ratio)
        rho_seg = rho[self.vsl]
        Veq_seg = engine.links.Veq(rho_seg, v_star, rho_star, a_star)
        Veq_new = Veq_default
        Veq_new[self.vsl] = Veq_seg
        return Veq_new
    