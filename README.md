This is a project to experiment with Python, Fast Fourier Transforms, and audio visualisation on the Raspberry Pi.

### Docker

You can run this in a (privileged) Docker container:

```bash
# Build the image:
docker build -t audio_levels .

# Run via a privileged container (for SPI access)
docker run --privileged audio_levels:latest
```
