
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

model = PingNet(exc_input_func=np.sin,
                n_outputs=n_out,
                n_excitatory=n_exc,
                n_inhibitory=n_inh,
                
                w_initial_I = -4*w_unit, # baseline inhibitory synaptic weight
                w_initial_E =  10*w_unit, # baseline excitatory synaptic weight
                w_initial_EI =  4*w_unit, # baseline feedback inhibition synaptic weight
                w_initial_EE =  15*w_unit, # baseline recurrent excitatory synaptic weight
                w_initial_E_Fb =  4*w_unit, # baseline output to excitatory synaptic weight (when connect_exc_fb = True)
                w_II = -4*w_unit,
                w_EI_Ext = 3*w_unit,
                
                connect_exc_inh_input = True,
                connect_inh_inh_input = True,
                connect_exc_fb = True,
)


with model:
    p_input = nengo.Probe(model.exc_input)
    p_exc = nengo.Probe(model.exc, synapse=0.01)
    p_exc_neurons = nengo.Probe(model.exc.neurons, synapse=0.01)
    p_inh_neurons = nengo.Probe(model.inh.neurons, synapse=0.01)
    p_out = nengo.Probe(model.out, synapse=0.01)
    p_out_neurons = nengo.Probe(model.out.neurons, synapse=0.01)
    p_inh_weights = nengo.Probe(model.conn_I, 'weights', sample_every=1.)
    p_exc_weights = nengo.Probe(model.conn_E, 'weights', sample_every=1.)
    p_rec_weights = nengo.Probe(model.conn_EE, 'weights', sample_every=1.)

with nengo.Simulator(model) as sim:
    sim.run(15)
    
plt.figure()
plt.plot(sim.trange(), sim.data[p_input], label="Input")
plt.plot(sim.trange(), sim.data[p_exc], label="Decoded output of exc")
plt.plot(sim.trange(), sim.data[p_out], label="Decoded output")
plt.legend()

plt.figure()
exc_spikes = sim.data[p_exc_neurons]
exc_rates = rates_kernel(sim.trange(), exc_spikes, tau=0.1)
plt.imshow(exc_rates.T, interpolation="nearest", aspect="auto")
plt.xlabel("Time [ms]")
plt.ylabel("Neuron number")
plt.colorbar();

plt.figure()
inh_spikes = sim.data[p_inh_neurons]
inh_rates = rates_kernel(sim.trange(), inh_spikes, tau=0.1)
plt.imshow(inh_rates.T, interpolation="nearest", aspect="auto")
plt.xlabel("Time [ms]")
plt.ylabel("Neuron number")
plt.colorbar();

plt.figure()
output_spikes = sim.data[p_out_neurons]
output_rates = rates_kernel(sim.trange(), output_spikes, tau=0.1)
plt.imshow(output_rates.T, interpolation="nearest", aspect="auto")
plt.xlabel("Time [ms]")
plt.ylabel("Neuron number")
plt.colorbar();

plt.show()

