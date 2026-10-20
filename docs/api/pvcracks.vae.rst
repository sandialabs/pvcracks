pvcracks.vae subpackage
========================

The current model is a deterministic skip-connected 3-channel autoencoder in
the historical ``vae`` namespace. It reconstructs crack/busbar/dark masks and
exposes a 320-dimensional pooled embedding. Its checkpoint ships with the
package; older VAE classes remain available for historical imports.

.. automodule:: pvcracks.vae
   :members:
   :undoc-members:
   :show-inheritance:

VAE helper modules
------------------

.. automodule:: pvcracks.vae.VAE_3CH_skipconn
   :members:
   :undoc-members:

.. automodule:: pvcracks.vae.inference
   :members:
   :undoc-members:

.. automodule:: pvcracks.vae.features
   :members:
   :undoc-members:
