# Homodyned K distribution
- [seabra2011rf](/papers/seabra2008modeling.pdf):
    - Finally, a study recently presented in [29] compared the compression parameter estimation of the well-established method proposed in [7,8](/papers/Homodyned%20K/prager2003decompression.pdf) with the approach described in this chapter, observing that the latter provides better results in terms of parameter estimation accuracy. As pointed out in [29](/papers/Homodyned%20K/paskas2009two.pdf) this could be explained as the decompression method proposed in [this chapter](/papers/seabra2008modeling.pdf) is based on the statistics for the compressed signal, while the approach presented in [7,8](/papers/Homodyned%20K/prager2003decompression.pdf) uses statistics for the uncompressed signal, and attempts to match theoretically calculated normalized moments with those determined directly from the image. **The process of fitting the moments calculated in the image with theoretical moments of the exponential distribution (cf. [8](/papers/Homodyned%20K/prager2003decompression.pdf)) is extremely sensitive to the order of the moment n, and this could create uncertainty on the decompression parameter to be estimated.**
        - [this chapter] [seabra2008modeling](/papers/seabra2008modeling.pdf), [seabra2001rf](/papers/seabra2011rf.pdf)
        - [8] [prager2003decompression](/papers/Homodyned%20K/prager2003decompression.pdf)
        - [29] [paskas2009two](/papers/Homodyned%20K/paskas2009two.pdf)
        - [Implementación con data sintética](../code/estimacion_seabra_data_generada.ipynb)
        - [Implementación con data real](../code/estimacion_seabra_data_real.ipynb)
- **Quantitative Ultrasound in Soft Tissues**
    - **1.2.2 Envelope stats techniques**
        - Some of the distributions considered in these models include the Rayleigh, the Rician, the K, the homodyned-K, and the Nakagami distributions; these distri-butions have been described extensively in the literature.
            - [**Destrempes and Cloutier 2010 - A CRITICAL REVIEW AND UNIFORMIZED REPRESENTATION OF STATISTICAL DISTRIBUTIONS MODELING THE ULTRASOUND ECHO ENVELOPE**](/papers/Homodyned%20K/destrempes2010critical.pdf)
                
                **Abstract:** We conclude that the homodyned K-distribution is the only model among the literature for which the parameters have a **physical meaning** that is consistent with the limiting case, although the other distributions may fit real data.
                
                The homodyned K-distribution was first introduced and studied (Jakeman 1980; Jakeman and Tough 1987) in the context of random walks viewed as a model of weak scattering. Thus, the K-distribution is a special case of the homodyned K-distribution, and the Rayleigh and the Rice distributions are limiting cases of the two former distributions (namely, the effective density of random scatterers is ‘‘infinite’’).
                
                **Introduction:**
                
                In Prager et al. (2003), a decompression algorithm is proposed, assuming the homodyned K-distribution for the envelope.
                
                - [Decompression and speckle detection for ultrasound images using the homodyned k-distribution](/papers/Homodyned%20K/prager2003decompression.pdf)
                    - [Implementando (sale mal)](../code/estimacion_prager_data_generada.ipynb)
        - [Dutt and Greenleaf 1994 - Ultrasound echo envelope analysis using a homodyned K distribution signal model](/papers/Homodyned%20K/dutt1994ultrasound.pdf)
            - Probability distribution function of the amplitude of the echo envelope ([used to generate data](../code/hom_k_dist_gen.m)):
                
                $p_A(A) = A \int_0^{\infty} x J_0(sx) J_0(Ax) \left( 1 + \frac{x^2 \sigma^2}{2 \mu} \right)^{-\mu} \, dx,$

                where $J_0(\cdot)$ is the zeroth order Bessel function of the first kind, $s^2$ is the coherent signal energy, $\sigma^2$ is the diffuse signal energy, and $\mu$ is the same parameter as defined in the K distribution. The derived parameter $k = \frac{s}{\sigma}$ is the ratio of the coherent to diffuse signal and can be used to describe the level of structure or periodicity in scatterer locations.

        - [Hruska and Oelze 2009 - Improved Parameter Estimates Based on the Homodyned K Distribution](/papers/Homodyned%20K/hruska2009improved.pdf)
            - Only estimates $\mu$ and $k$.
    - **7. Review of envelope statistics models for quantitative ultrasound imaging and tissue characterization**
        - The main assumptions made are **(1) the absence of log-compression** or application of nonlinear filtering on the echo envelope of the radiofrequency signal and (2) the randomness and independence of the diffuse scatterers.
        - The (two-dimensional) homodyned K-distribution (Jakeman 1980; Jakeman and Tough 1987) is defined by
            
            $P_{\text{HK}}(A \mid \varepsilon, \sigma^2, \alpha) = A \int_0^{\infty} u J_0(u \varepsilon) J_0(u A) \left( 1 + \frac{u^2 \sigma^2}{2} \right)^{-\alpha} \, du,$
          
          where σ2 > 0, α > 0, ε ≥ 0, and J0 denotes the Bessel function of the first kind of order 0. In Jakeman and Tough (1987, Eq. 4.13), the homodyned K-distribution is expressed in terms of the parameters α, a2 = nσ2α, and a0 = ε, in the context of n-dimensional random walks.