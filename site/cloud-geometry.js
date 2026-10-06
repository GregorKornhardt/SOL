// Equal axis scales keep an orthogonal projection perpendicular on screen.
export function cloudFrame(width, height) {
  const scale = Math.min((width - 60) / 5.1, (height - 76) / 3.9);
  const origin = [36 + (width - 60) / 2 - .65 * scale, (height - 76) / 2 + 38 + .05 * scale];
  return { scale, point: ([x, y]) => [origin[0] + x * scale, origin[1] - y * scale] };
}
