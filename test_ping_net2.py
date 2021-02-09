
import matplotlib.pyplot as plt
import numpy as np
from ping_net import PingNet
import nengo
from nengo.processes import Piecewise
from nengo_extras.plot_spikes import (
    cluster, merge, plot_spikes, preprocess_spikes, sample_by_variance)
from nengo_extras.neurons import (
    rates_kernel, rates_isi, spikes2events )

n_exc = 500
n_inh = 100
n_out = 200

w_unit = 0.0001

model = nengo.Network()
with model:
    m1 = PingNet(exc_input_func=np.sin,
                n_outputs=n_out,
                n_excitatory=n_exc,
                n_inhibitory=n_inh,
                
                w_initial_I = -4*w_unit, # baseline inhibitory synaptic weight
                w_initial_E =  10*w_unit, # baseline excitatory synaptic weight
                w_initial_EI =  4*w_unit, # baseline feedback inhibition synaptic weight
                w_initial_EE =  25*w_unit, # baseline recurrent excitatory synaptic weight
                w_initial_E_Fb =  2*w_unit, # baseline output to excitatory synaptic weight (when connect_exc_fb = True)
                w_II = -4*w_unit,
                w_EI_Ext = 2*w_unit,
            
                connect_exc_inh_input = True,
                connect_inh_inh_input = True,
                connect_exc_fb = True)

    m2 = PingNet(
                n_outputs=n_out,
                n_excitatory=n_exc,
                n_inhibitory=n_inh,
                
                w_initial_I = -4*w_unit, # baseline inhibitory synaptic weight
                w_initial_E =  10*w_unit, # baseline excitatory synaptic weight
                w_initial_EI =  4*w_unit, # baseline feedback inhibition synaptic weight
                w_initial_EE =  25*w_unit, # baseline recurrent excitatory synaptic weight
                w_initial_E_Fb =  2*w_unit, # baseline output to excitatory synaptic weight (when connect_exc_fb = True)
                w_II = -4*w_unit,
                w_EI_Ext = 2*w_unit,
                
                connect_exc_inh_input = True,
                connect_inh_inh_input = True,
                connect_exc_fb = True)

    nengo.Connection(m1.out.neurons, m2.exc.neurons,
                     transform=np.where(np.random.uniform(0, 1, size=(n_exc, n_out))<0.65,
                                        np.random.normal(size=(n_exc, n_out))*20*w_unit, 0))
    nengo.Connection(m1.out.neurons, m2.inh.neurons,
                     transform=np.random.uniform(0, 4*w_unit, size=(n_inh, n_out, )))



with model:
    p_input = nengo.Probe(m1.exc_input)
    p_exc1 = nengo.Probe(m1.exc, synapse=0.01)
    p_exc1_neurons = nengo.Probe(m1.exc.neurons, synapse=0.01)
    p_inh1 = nengo.Probe(m1.inh, synapse=0.01)
    p_out1 = nengo.Probe(m1.out, synapse=0.01)
    p_out1_neurons = nengo.Probe(m1.out.neurons, synapse=0.01)
    p_exc2 = nengo.Probe(m2.exc, synapse=0.01)
    p_exc2_neurons = nengo.Probe(m2.exc.neurons, synapse=0.01)
    p_inh2 = nengo.Probe(m2.inh, synapse=0.01)
    p_out2 = nengo.Probe(m2.out, synapse=0.01)
    p_out2_neurons = nengo.Probe(m2.out.neurons, synapse=0.01)

with nengo.Simulator(model) as sim:
    sim.run(15)
    
plt.figure()
plt.plot(sim.trange(), sim.data[p_input], label="Input")
plt.plot(sim.trange(), sim.data[p_exc1], label="Decoded output of exc1")
plt.plot(sim.trange(), sim.data[p_out1], label="Decoded output1")
plt.legend()

plt.figure()
plt.plot(sim.trange(), sim.data[p_input], label="Input")
plt.plot(sim.trange(), sim.data[p_exc2], label="Decoded output of exc2")
plt.plot(sim.trange(), sim.data[p_out2], label="Decoded output2")
plt.legend()

plt.figure()
output1_spikes = sim.data[p_out1_neurons]
output1_rates = rates_kernel(sim.trange(), output1_spikes, tau=0.1)
plt.imshow(output1_rates.T, interpolation="nearest", aspect="auto")
plt.xlabel("Time [ms]")
plt.ylabel("Neuron number")
plt.colorbar();

plt.figure()
output2_spikes = sim.data[p_out2_neurons]
output2_rates = rates_kernel(sim.trange(), output2_spikes, tau=0.1)
plt.imshow(output2_rates.T, interpolation="nearest", aspect="auto")
plt.xlabel("Time [ms]")
plt.ylabel("Neuron number")
plt.colorbar();

plt.show()

