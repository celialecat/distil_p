"use client";

import { useEffect, useRef } from "react";
import * as THREE from "three";

export type LiquidShaderState = "idle" | "running" | "complete";

type LiquidShaderProps = {
  state?: LiquidShaderState;
  centerDim?: number;
};

const palettes: Record<LiquidShaderState, [string, string, string]> = {
  idle: ["#07545f", "#8f4d12", "#102a48"],
  running: ["#00bfa6", "#e88927", "#4535a8"],
  complete: ["#1bc59b", "#f1a53b", "#a34fe7"],
};

const stateEnergy: Record<LiquidShaderState, number> = {
  idle: 0.82,
  running: 1.18,
  complete: 1.08,
};

const vertexShader = `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = vec4(position, 1.0);
  }
`;

const fragmentShader = `
  precision highp float;
  uniform float uTime;
  uniform vec2 uResolution;
  uniform vec3 uPaletteA;
  uniform vec3 uPaletteB;
  uniform vec3 uPaletteC;
  uniform float uCenterDim;
  uniform float uEnergy;
  varying vec2 vUv;

  float hash(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
  }

  float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), f.x),
      mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), f.x), f.y);
  }

  float fbm(vec2 p) {
    float value = 0.0;
    float amplitude = 0.5;
    for (int i = 0; i < 5; i++) {
      value += amplitude * noise(p);
      p = p * 2.02 + vec2(17.3, 9.1);
      amplitude *= 0.5;
    }
    return value;
  }

  void main() {
    vec2 uv = vUv;
    float aspect = uResolution.x / max(uResolution.y, 1.0);
    vec2 p = (uv - 0.5) * vec2(aspect, 1.0);
    float t = uTime * 0.055;
    vec2 flow = p * 1.6 + vec2(t * 0.25, -t * 0.16);
    float field = fbm(flow + fbm(flow * 1.4 + t));
    float wave = sin((p.x + field * 0.9) * 4.0 + t) * 0.5 + 0.5;
    float glow = smoothstep(1.15, 0.05, length(p * vec2(0.75, 1.0)));
    vec3 liquid = mix(uPaletteA, uPaletteB, smoothstep(0.2, 0.85, field));
    liquid = mix(liquid, uPaletteC, smoothstep(0.45, 0.95, wave) * 0.66);
    float dim = mix(1.0, smoothstep(0.0, 0.9, length(p)), uCenterDim);
    float vignette = smoothstep(1.3, 0.25, length(p));
    gl_FragColor = vec4(liquid * (0.28 + glow * 0.62) * uEnergy * dim * vignette, 1.0);
  }
`;

export default function LiquidShader({ state = "idle", centerDim = 0.8 }: LiquidShaderProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: false, powerPreference: "low-power" });
    } catch {
      return;
    }

    const scene = new THREE.Scene();
    const camera = new THREE.Camera();
    const geometry = new THREE.PlaneGeometry(2, 2);
    const colors = palettes[state];
    const material = new THREE.ShaderMaterial({
      uniforms: {
        uTime: { value: 0 },
        uResolution: { value: new THREE.Vector2() },
        uPaletteA: { value: new THREE.Color(colors[0]) },
        uPaletteB: { value: new THREE.Color(colors[1]) },
        uPaletteC: { value: new THREE.Color(colors[2]) },
        uCenterDim: { value: centerDim },
        uEnergy: { value: stateEnergy[state] },
      },
      vertexShader,
      fragmentShader,
    });
    scene.add(new THREE.Mesh(geometry, material));

    const resize = () => {
      const width = window.innerWidth;
      const height = window.innerHeight;
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
      renderer.setSize(width, height, false);
      material.uniforms.uResolution.value.set(width, height);
    };
    resize();
    window.addEventListener("resize", resize);

    let frame = 0;
    const animate = (time: number) => {
      material.uniforms.uTime.value = time;
      renderer.render(scene, camera);
      frame = window.requestAnimationFrame(animate);
    };
    frame = window.requestAnimationFrame(animate);

    return () => {
      window.cancelAnimationFrame(frame);
      window.removeEventListener("resize", resize);
      geometry.dispose();
      material.dispose();
      renderer.dispose();
    };
  }, [centerDim, state]);

  return <canvas ref={canvasRef} aria-hidden="true" className="liquid-shader absolute inset-0 h-full w-full" />;
}
