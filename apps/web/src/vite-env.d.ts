/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_TARGET?: 'device';
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
