/** Record id. Hermes on iOS has no `crypto.randomUUID`, so this cannot depend on it. */
export function newId(): string {
  const rand = () => Math.floor(Math.random() * 0x100000000).toString(16).padStart(8, '0');
  return `${Date.now().toString(16)}-${rand()}-${rand()}`;
}
