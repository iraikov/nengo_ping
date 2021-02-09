
import numpy as np
import nengo

## Pyramidal-interneuron-gamma model (PING) motif implementation. 
class PingNet(nengo.Network):
    def __init__(self,
                 exc_input_func = None,
                 inh_input_func = None,
                 dimensions = 1,
                 n_outputs = 200,
                 n_inhibitory = 100,
                 n_excitatory = 500,
                 w_input = 1, # external input weights
                 w_initial_I = -1e-2, # baseline inhibitory synaptic weight
                 w_initial_E =  1e-2, # baseline excitatory synaptic weight
                 w_initial_EI =  1e-3, # baseline feedback inhibition synaptic weight
                 w_initial_EE =  1e-3, # baseline recurrent excitatory synaptic weight
                 w_initial_E_Fb =  1e-3, # baseline output to excitatory synaptic weight (when connect_exc_fb = True)
                 w_EI_Ext = 1e-3, # weight of excitatory connection to inhibitory inputs (when connect_exc_inh_input = True)
                 w_II = -1e-3, # weight of inhibitory to inhibitory connections (when connect_inh_inh_input = True)
                 p_E = 0.1, # uniform probability of connection of excitatory inputs to outputs
                 p_EI = 0.25, # uniform probability of feedback connections to inhibitory cells
                 p_EE = 0.05, # uniform probability of recurrent connections
                 p_EI_Ext = 0.15, # uniform probability of excitatory connection to inhibitory inputs (when connect_exc_inh_input = True)
                 p_E_Fb = 0.05, # uniform probability of outputs to excitatory inputs (when connect_exc_fb = True)
                 p_II = 0.1, # uniform probability of inhibitory to inhibitory inputs (when connect_inh_inh = True)
                 tau_I = 0.008, # filter for inhibitory inputs
                 tau_E = 0.006, # filter for excitatory inputs
                 tau_EI = 0.006, # filter for feedback inhibitory connections
                 tau_EE = 0.006, # filter for recurrent connections
                 tau_EI_Ext = 0.006, # filter for excitatory connection to inhibitory inputs (when connect_exc_inh_input = True)
                 tau_E_Fb = 0.006, # filter for output connection to excitatory inputs (when connect_exc_fb  = True)
                 tau_input = 0.005, # filter for node input
                 connect_exc_inh_input = False,
                 connect_inh_inh_input = False,
                 connect_exc_fb = False,
                 label = None,
                 seed = 0,
                 add_to_container = None,
                 weights_I = None,
                 weights_E = None,
                 weights_EE = None,
                 weights_E_Fb = None,
                 **kwds
                 ):
        super().__init__(label, seed, add_to_container)
        
        self.dimensions = dimensions
        self.n_excitatory = n_excitatory
        self.n_inhibitory = n_inhibitory
        self.n_outputs = n_outputs
        
        rng = np.random.RandomState(seed=seed)

        assert(w_initial_I < 0)
        assert(w_initial_E > 0)
        assert(w_initial_EI > 0)
        assert(w_initial_EE > 0)
        assert(w_initial_E_Fb > 0)

        with self:

            self.exc_input = None
            self.inh_input = None
            if exc_input_func is not None:
                self.exc_input = nengo.Node(output=exc_input_func)
            if inh_input_func is not None:
                self.inh_input = nengo.Node(output=inh_input_func)
                
            with self.exc_ens_config:

                self.exc = nengo.Ensemble(self.n_excitatory, dimensions=self.dimensions)

            with self.inh_ens_config:
            
                self.inh = nengo.Ensemble(self.n_inhibitory, dimensions=self.dimensions)
            
            with self.out_ens_config:
                self.out = nengo.Ensemble(self.n_outputs, dimensions=self.dimensions)

            if self.exc_input is not None:
                nengo.Connection(self.exc_input, self.exc,
                                 synapse=nengo.Alpha(tau_input),
                                 transform=[w_input])
            
            if self.inh_input is not None:
                nengo.Connection(self.inh_input, self.inh,
                                 synapse=nengo.Alpha(tau_input),
                                 transform=[w_input])

            if connect_exc_inh_input and (self.exc_input is not None):
                weights_initial_EI_Ext = np.random.uniform(0, w_EI_Ext, size=(n_inhibitory, n_excitatory))
                nengo.Connection(self.exc.neurons, self.inh.neurons,
                                 synapse=nengo.Alpha(tau_EI_Ext),
                                 transform=weights_initial_EI_Ext)
                
            

            if weights_I is not None:
                weights_initial_I = weights_I
            else:
                weights_initial_I = np.random.uniform(0, w_initial_I, size=(n_outputs, n_inhibitory))
            self.conn_I = nengo.Connection(self.inh.neurons,
                                           self.out.neurons,
                                           transform=weights_initial_I,
                                           synapse=nengo.Alpha(tau_I))
            self.conn_II = None
            if connect_inh_inh_input:
                weights_initial_II = np.random.uniform(0, w_II, size=(n_inhibitory, n_inhibitory))
                self.conn_II = nengo.Connection(self.inh.neurons,
                                                self.inh.neurons,
                                                transform=weights_initial_II,
                                                synapse=nengo.Alpha(tau_I))

                
            if weights_E is not None:
                weights_initial_E = weights_E
            else:
                weights_initial_E = np.where(np.random.uniform(0, 1, size=(n_outputs, n_excitatory))<p_E,
                                             np.random.normal(size=(n_outputs, n_excitatory))*w_initial_E, 0)
                
            self.conn_E = nengo.Connection(self.exc.neurons,
                                           self.out.neurons, 
                                           transform=weights_initial_E,
                                           synapse=nengo.Alpha(tau_E))

            weights_initial_EI = np.random.uniform(0, w_initial_EI, size=(n_inhibitory, n_outputs))
            self.conn_EI = nengo.Connection(self.out.neurons,
                                            self.inh.neurons,
                                            transform=weights_initial_EI,
                                            synapse=nengo.Alpha(tau_EI))

            self.conn_E_Fb = None
            if connect_exc_fb:
                if weights_E_Fb is not None:
                    weights_initial_E_Fb = weights_E_Fb
                else:
                    weights_initial_E_Fb = np.where(np.random.uniform(0, 1, size=(n_excitatory, n_outputs))<p_E_Fb,
                                                    np.random.normal(size=(n_excitatory, n_outputs))*w_initial_E_Fb, 0)
                    self.conn_E_Fb = nengo.Connection(self.out.neurons,
                                                      self.exc.neurons, 
                                                      transform=weights_initial_E_Fb,
                                                      synapse=nengo.Alpha(tau_E_Fb))
                
            
            if self.n_excitatory > 1:
                weights_initial_EE = np.where(np.random.uniform(0, 1, size=(n_excitatory, n_excitatory))<p_EE,
                                              np.random.normal(size=(n_excitatory, n_excitatory))*w_initial_EE, 0)

                self.conn_EE = nengo.Connection(self.exc.neurons, 
                                                self.exc.neurons, 
                                                transform=weights_initial_EE,
                                                synapse=nengo.Alpha(tau_EE))
            else:
                self.conn_EE = None

                             
    @property
    def exc_ens_config(self):
        """(Config) Defaults for excitatory input ensemble creation."""
        cfg = nengo.Config(nengo.Ensemble, nengo.Connection)
        cfg[nengo.Ensemble].update(
            {
            "radius": 1,
            "max_rates": nengo.dists.Choice([20]),
            }
            )
        cfg[nengo.Connection].synapse = None
        return cfg
    
    @property
    def inh_ens_config(self):
        """(Config) Defaults for inhibitory input ensemble creation."""
        cfg = nengo.Config(nengo.Ensemble, nengo.Connection)
        cfg[nengo.Ensemble].update(
            {
            "radius": 1,
            "max_rates": nengo.dists.Choice([40])
            }
            )
        cfg[nengo.Connection].synapse = None
        return cfg
    
    @property
    def out_ens_config(self):
        """(Config) Defaults for excitatory input ensemble creation."""
        cfg = nengo.Config(nengo.Ensemble, nengo.Connection)
        cfg[nengo.Ensemble].update(
            {
            "neuron_type": nengo.LIF(),
            "radius": 10,
            "max_rates": nengo.dists.Choice([40]),
            "intercepts": nengo.dists.Choice([0.001]*self.dimensions),
            }
            )
        cfg[nengo.Connection].synapse = None
        return cfg
