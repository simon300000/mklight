// The layout's 3-D view in WebGL2 (the macOS app draws it with Metal):
// polylines as lines (profiles, rims and rings, edges, stop, image, rays,
// axis), the engine's optional surface mesh flat shaded (the selected surface
// in the accent), and beam footprints filled translucently. Orthographic,
// with the camera of camera.ts; positions are relative to the scene's centre
// so float32 keeps them precise.

import { seriesColor } from '@/lib/theme'
import { rotation, type Camera, type Size } from './camera'
import { doeColor, rayColor, rgba } from './draw'
import { KIND, type Footprint, type Scene } from './scene'

const LINE_VS = `#version 300 es
uniform mat3 uR; uniform vec3 uC; uniform vec3 uS;
in vec3 aPos; in vec4 aColor; out vec4 vColor;
void main() { vec3 v = uR * aPos - uC; gl_Position = vec4(v.x * uS.x, v.y * uS.y, -v.z * uS.z, 1.0); vColor = aColor; }`
const LINE_FS = `#version 300 es
precision mediump float; in vec4 vColor; out vec4 o;
void main() { o = vec4(vColor.rgb * vColor.a, vColor.a); }`
const MESH_VS = `#version 300 es
uniform mat3 uR; uniform vec3 uC; uniform vec3 uS;
in vec3 aPos; in vec3 aNormal; in vec4 aColor; out vec4 vColor;
void main() {
  vec3 v = uR * aPos - uC;
  gl_Position = vec4(v.x * uS.x, v.y * uS.y, -v.z * uS.z, 1.0);
  vec3 n = normalize(uR * aNormal);
  float shade = 0.35 + 0.65 * abs(n.z);
  vColor = vec4(aColor.rgb * shade, aColor.a);
}`

interface Batch {
  vao: WebGLVertexArrayObject
  buffers: WebGLBuffer[]
  count: number
}

export interface GlOptions {
  dark: boolean
  selection: number
  coloring: 'field' | 'wavelength'
  colors: { label: string; secondary: string; accent: string }
  footprints: Footprint[]
}

function compile(gl: WebGL2RenderingContext, vs: string, fs: string): WebGLProgram {
  const p = gl.createProgram()!
  for (const [type, src] of [
    [gl.VERTEX_SHADER, vs],
    [gl.FRAGMENT_SHADER, fs],
  ] as const) {
    const s = gl.createShader(type)!
    gl.shaderSource(s, src)
    gl.compileShader(s)
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) ?? 'shader')
    gl.attachShader(p, s)
  }
  gl.linkProgram(p)
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p) ?? 'link')
  return p
}

export class Gl3D {
  private gl: WebGL2RenderingContext
  private line: WebGLProgram
  private mesh: WebGLProgram
  private lines: Batch | null = null
  private triangles: Batch | null = null
  private footLines: Batch | null = null
  private footFill: Batch | null = null
  private origin: [number, number, number] = [0, 0, 0]
  private radius = 1

  /** Throws when the browser has no WebGL2. */
  constructor(canvas: HTMLCanvasElement) {
    const gl = canvas.getContext('webgl2', { antialias: true, premultipliedAlpha: true, alpha: true })
    if (!gl) throw new Error('This browser has no WebGL2.')
    this.gl = gl
    this.line = compile(gl, LINE_VS, LINE_FS)
    this.mesh = compile(gl, MESH_VS, LINE_FS)
  }

  private batch(prog: WebGLProgram, attrs: { name: string; data: Float32Array; size: number }[], count: number): Batch {
    const gl = this.gl
    const vao = gl.createVertexArray()!
    gl.bindVertexArray(vao)
    const buffers: WebGLBuffer[] = []
    for (const a of attrs) {
      const b = gl.createBuffer()!
      gl.bindBuffer(gl.ARRAY_BUFFER, b)
      gl.bufferData(gl.ARRAY_BUFFER, a.data, gl.STATIC_DRAW)
      const loc = gl.getAttribLocation(prog, a.name)
      if (loc >= 0) {
        gl.enableVertexAttribArray(loc)
        gl.vertexAttribPointer(loc, a.size, gl.FLOAT, false, 0, 0)
      }
      buffers.push(b)
    }
    gl.bindVertexArray(null)
    return { vao, buffers, count }
  }

  private free(b: Batch | null) {
    if (!b) return
    for (const x of b.buffers) this.gl.deleteBuffer(x)
    this.gl.deleteVertexArray(b.vao)
  }

  /** Builds the buffers of a scene (and the colours that depend on the options). */
  setScene(s: Scene, o: GlOptions) {
    this.free(this.lines)
    this.free(this.triangles)
    this.free(this.footLines)
    this.free(this.footFill)
    const b = s.bounds
    this.origin = [0, 1, 2].map((k) => {
      const c = ((b.min[k] ?? 0) + (b.max[k] ?? 0)) / 2
      return Number.isFinite(c) ? c : 0
    }) as [number, number, number]
    const ext = [0, 1, 2].map((k) => (b.max[k] ?? 0) - (b.min[k] ?? 0)).filter(Number.isFinite)
    this.radius = Math.max(Math.hypot(...ext), 1)
    const [ox, oy, oz] = this.origin
    const label = rgba(o.colors.label), secondary = rgba(o.colors.secondary), accent = rgba(o.colors.accent), doe = rgba(doeColor(o.dark))

    // Lines.
    const pos: number[] = [], col: number[] = []
    const { xyz, offsets, kinds, flags, surfaces } = s
    for (let p = 0; p < kinds.length; p++) {
      const k = kinds[p]
      if (k === KIND.outline) continue
      let c: [number, number, number, number]
      if (k === KIND.ray) {
        const rc = rgba(rayColor(s, p, o.coloring, o.dark))
        c = [rc[0], rc[1], rc[2], flags[p] & 2 ? 0.2 : 0.75]
      } else if (k === KIND.axis) c = [secondary[0], secondary[1], secondary[2], 0.45]
      else if (k === KIND.stop) c = accent
      else if (surfaces[p] >= 0 && surfaces[p] === o.selection) c = accent
      else if (flags[p] & 4) c = doe
      else c = [label[0], label[1], label[2], k === KIND.rim ? 0.55 : 0.9]
      for (let i = offsets[p]; i + 1 < offsets[p + 1]; i++) {
        const a = 3 * i, e = 3 * (i + 1)
        if (![xyz[a], xyz[a + 1], xyz[a + 2], xyz[e], xyz[e + 1], xyz[e + 2]].every(Number.isFinite)) continue
        pos.push(xyz[a] - ox, xyz[a + 1] - oy, xyz[a + 2] - oz, xyz[e] - ox, xyz[e + 1] - oy, xyz[e + 2] - oz)
        col.push(...c, ...c)
      }
    }
    this.lines = this.batch(this.line, [{ name: 'aPos', data: new Float32Array(pos), size: 3 }, { name: 'aColor', data: new Float32Array(col), size: 4 }], pos.length / 3)

    // Surface mesh, flat shaded.
    this.triangles = null
    if (s.mesh) {
      const m = s.mesh.xyz
      const t = s.mesh.surfaces.length
      const mp = new Float32Array(t * 9), mn = new Float32Array(t * 9), mc = new Float32Array(t * 12)
      const base: [number, number, number, number] = o.dark ? [0.62, 0.72, 0.88, 0.42] : [0.5, 0.6, 0.78, 0.38]
      const sel: [number, number, number, number] = [accent[0], accent[1], accent[2], 0.6]
      for (let k = 0; k < t; k++) {
        const a = k * 9
        const ux = m[a + 3] - m[a], uy = m[a + 4] - m[a + 1], uz = m[a + 5] - m[a + 2]
        const vx = m[a + 6] - m[a], vy = m[a + 7] - m[a + 1], vz = m[a + 8] - m[a + 2]
        let nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx
        const l = Math.hypot(nx, ny, nz) || 1
        nx /= l
        ny /= l
        nz /= l
        const c = s.mesh.surfaces[k] === o.selection ? sel : base
        for (let v = 0; v < 3; v++) {
          mp[a + 3 * v] = m[a + 3 * v] - ox
          mp[a + 3 * v + 1] = m[a + 3 * v + 1] - oy
          mp[a + 3 * v + 2] = m[a + 3 * v + 2] - oz
          mn[a + 3 * v] = nx
          mn[a + 3 * v + 1] = ny
          mn[a + 3 * v + 2] = nz
          mc.set(c, k * 12 + 4 * v)
        }
      }
      this.triangles = this.batch(
        this.mesh,
        [
          { name: 'aPos', data: mp, size: 3 },
          { name: 'aNormal', data: mn, size: 3 },
          { name: 'aColor', data: mc, size: 4 },
        ],
        t * 3,
      )
    }

    // Footprints: a translucent fill (a fan of the convex outline) and its edge.
    this.footLines = this.footFill = null
    if (o.footprints.length) {
      const fp: number[] = [], fc: number[] = [], lp: number[] = [], lc: number[] = []
      for (const f of o.footprints) {
        const c = rgba(seriesColor(f.field, o.dark))
        const fill = [c[0], c[1], c[2], 0.28], edge = [c[0], c[1], c[2], 0.9]
        const n = f.points.length / 3
        const P = (i: number) => [f.points[3 * i] - ox, f.points[3 * i + 1] - oy, f.points[3 * i + 2] - oz]
        for (let i = 1; i + 1 < n; i++) {
          fp.push(...P(0), ...P(i), ...P(i + 1))
          fc.push(...fill, ...fill, ...fill)
        }
        for (let i = 0; i < n; i++) {
          lp.push(...P(i), ...P((i + 1) % n))
          lc.push(...edge, ...edge)
        }
      }
      this.footFill = this.batch(this.line, [{ name: 'aPos', data: new Float32Array(fp), size: 3 }, { name: 'aColor', data: new Float32Array(fc), size: 4 }], fp.length / 3)
      this.footLines = this.batch(this.line, [{ name: 'aPos', data: new Float32Array(lp), size: 3 }, { name: 'aColor', data: new Float32Array(lc), size: 4 }], lp.length / 3)
    }
  }

  draw(cam: Camera, size: Size, dpr: number) {
    const gl = this.gl
    const w = Math.max(1, Math.round(size.w * dpr)), h = Math.max(1, Math.round(size.h * dpr))
    if (gl.canvas.width !== w || gl.canvas.height !== h) {
      gl.canvas.width = w
      gl.canvas.height = h
    }
    gl.viewport(0, 0, w, h)
    gl.clearColor(0, 0, 0, 0)
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)
    const r = rotation(cam, '3d')
    // Column-major for GLSL.
    const R = new Float32Array([r[0], r[3], r[6], r[1], r[4], r[7], r[2], r[5], r[8]])
    const c = [cam.center[0] - this.origin[0], cam.center[1] - this.origin[1], cam.center[2] - this.origin[2]]
    const C = new Float32Array([r[0] * c[0] + r[1] * c[1] + r[2] * c[2], r[3] * c[0] + r[4] * c[1] + r[5] * c[2], r[6] * c[0] + r[7] * c[1] + r[8] * c[2]])
    const S = new Float32Array([(2 * cam.ppmm) / size.w, (2 * cam.ppmm) / size.h, 1 / (2 * this.radius)])
    const use = (p: WebGLProgram) => {
      gl.useProgram(p)
      gl.uniformMatrix3fv(gl.getUniformLocation(p, 'uR'), false, R)
      gl.uniform3fv(gl.getUniformLocation(p, 'uC'), C)
      gl.uniform3fv(gl.getUniformLocation(p, 'uS'), S)
    }
    gl.enable(gl.BLEND)
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA)
    gl.enable(gl.DEPTH_TEST)
    gl.depthFunc(gl.LEQUAL)
    // Lines write depth; the translucent mesh and footprints are drawn over them without.
    use(this.line)
    gl.depthMask(true)
    if (this.lines?.count) {
      gl.bindVertexArray(this.lines.vao)
      gl.drawArrays(gl.LINES, 0, this.lines.count)
    }
    gl.depthMask(false)
    if (this.triangles?.count) {
      use(this.mesh)
      gl.enable(gl.POLYGON_OFFSET_FILL)
      gl.polygonOffset(1, 1)
      gl.bindVertexArray(this.triangles.vao)
      gl.drawArrays(gl.TRIANGLES, 0, this.triangles.count)
      gl.disable(gl.POLYGON_OFFSET_FILL)
      use(this.line)
    }
    if (this.footFill?.count) {
      gl.bindVertexArray(this.footFill.vao)
      gl.drawArrays(gl.TRIANGLES, 0, this.footFill.count)
    }
    if (this.footLines?.count) {
      gl.bindVertexArray(this.footLines.vao)
      gl.drawArrays(gl.LINES, 0, this.footLines.count)
    }
    gl.depthMask(true)
    gl.bindVertexArray(null)
  }

  dispose() {
    this.free(this.lines)
    this.free(this.triangles)
    this.free(this.footLines)
    this.free(this.footFill)
    this.gl.deleteProgram(this.line)
    this.gl.deleteProgram(this.mesh)
    // Release the context once its canvas has left the page: browsers keep only
    // a few, and every remount of the layout (a tab, a separate window) makes a
    // new one. A canvas still in the page (React's development remount) keeps it.
    const gl = this.gl
    setTimeout(() => {
      if (!(gl.canvas as HTMLCanvasElement).isConnected) gl.getExtension('WEBGL_lose_context')?.loseContext()
    }, 0)
  }
}
