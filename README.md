# BMP-overlay

### Overlays one BMP image on top of another using AVX instructions.

## Usage

```sh
gcc bmp_main.c -o prog -mavx
```

```sh
./prog [background.bmp] [foreground.bmp] [res_name.bmp]
```
---
