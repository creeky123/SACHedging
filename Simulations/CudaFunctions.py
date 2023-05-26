from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32, xoroshiro128p_normal_float32
import math

## cuda functions for option valuation of both FIA and call
## calculating the call option value
@cuda.jit
def compute_heston_call(rng_states, s_0, v_0, theta, kappa, xi, rf, rho, dt, strike, shock, out, thread_split,
                        iterations):
    thread_id = cuda.grid(1)

    if thread_id > thread_split - 1:
        s_0 = s_0 + shock

        if thread_id > thread_split * 2 - 1:
            s_0 = s_0 - 2 * shock

    vol = v_0
    price = s_0
    sqrt_dt = math.sqrt(dt)
    for i in range(iterations):

        # euler scheme for heston
        # heston - s_t+1 = st * exp(r_f - v(t)/2)dt + sqrt(v(t)) * dW_S_t)
        # vol - v(t+1) = v(t) + kappa (theta - v(t))*dt + xi * sqrt(v(t)) * dW_V_t

        w_s = xoroshiro128p_normal_float32(rng_states, thread_id)
        w_v = xoroshiro128p_normal_float32(rng_states, thread_id)

        corr_random = rho * w_v + math.sqrt(1 - rho * rho) * w_s

        price = price * math.exp((rf - .5 * vol) * dt + math.sqrt(vol) * sqrt_dt * corr_random)

        vol = vol + kappa * dt * (theta - vol) + xi * math.sqrt(vol) * math.sqrt(dt) * w_v

        if vol < 0.01:
            vol = 0.01

    out[thread_id] = max(0, price - strike)

### calculating the option value of the FIA
@cuda.jit
def compute_heston_fia(rng_states, s_0, v_0, theta, kappa, xi, rf, rho, dt, strike, shock, out, thread_split,
                       iterations, pr_death, pr_lapse, lock_in, T):
    ##set s_0 outside of function with dim = #threads

    thread_id = cuda.grid(1)


    if thread_id > thread_split - 1:
        s_0 = s_0 + shock

        if thread_id > thread_split * 2 - 1:
            s_0 = s_0 - 2*shock

    price = s_0

    vol = v_0
    sqrt_dt = math.sqrt(dt)
    for i in range(iterations):
        w_s = xoroshiro128p_normal_float32(rng_states, thread_id)
        w_v = xoroshiro128p_normal_float32(rng_states, thread_id)

        corr_random = rho * w_v + math.sqrt(1 - rho * rho) * w_s

        price = price * math.exp((rf - .5 * vol) * dt + math.sqrt(vol) * sqrt_dt * corr_random)
        vol = vol + kappa * dt * (theta - vol) + xi * math.sqrt(vol) * math.sqrt(dt) * w_v

        if vol < 0.001:
            vol = 0.001
        # stocahstic death
        if xoroshiro128p_uniform_float32(rng_states, thread_id) > pr_death:
            out[thread_id] = max(0, 0.8 * (price - strike) * math.exp(-rf * T * (i / iterations)))  # death - pay 80% of gain
            return
        # stochastic lock in
        if lock_in and price > 1.5*strike: #ph locks in index price
            if xoroshiro128p_uniform_float32(rng_states, thread_id) > (1 / math.exp((price / strike) * 0.005)):
                out[thread_id] = max(0, (price - strike) * math.exp(-rf * T * (i / iterations)))
                return
        # stochastic lapse
        if xoroshiro128p_uniform_float32(rng_states, thread_id) > pr_lapse:
            out[thread_id] = 0
            return

    out[thread_id] = max(0, (price - strike) * math.exp(-rf * T))
