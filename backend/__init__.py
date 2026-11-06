"""Mirror's web backend: turns an uploaded photo or a product page into a glasses model.

Reconstruction reuses image_to_3d directly. Nothing here uses machine learning;
product photos are found with HTML metadata and chosen with OpenCV geometry.
"""
