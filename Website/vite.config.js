import { resolve } from "node:path";
import { defineConfig } from "vite";

const pages = ["index", "banana", "brain", "sdk", "sim"];

export default defineConfig({
  server: { open: true },
  build: {
    rollupOptions: {
      input: Object.fromEntries(pages.map(p => [p, resolve(import.meta.dirname, `${p}.html`)]))
    }
  }
});
