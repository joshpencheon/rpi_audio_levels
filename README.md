This is a project to experiment with Python, Fast Fourier Transforms, and audio visualisation on the Raspberry Pi.

![rpi_audio_levels](https://user-images.githubusercontent.com/30904/99916518-7de0c880-2d02-11eb-9d56-c3d84ddfe72d.gif)

### Docker

You can run this in a (privileged) Docker container:

```bash
# Build the image:
docker build -t audio_levels .

# Run via a privileged container (for SPI access)
docker run --privileged audio_levels:latest
```
