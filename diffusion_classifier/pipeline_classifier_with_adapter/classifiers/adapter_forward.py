def unet_with_adapters_forward(unet_with_adapters, latents, timesteps, text_embeds, extra_cond=None):
    """
    Forward-Funktion, kompatibel zu eval_error(...):
    Gibt direkt das Noise-Prediction-Tensor zurück (wie .sample).
    """
    return unet_with_adapters(
        latents,
        timesteps,
        encoder_hidden_states=text_embeds,
        extra_cond=extra_cond
    ).sample
