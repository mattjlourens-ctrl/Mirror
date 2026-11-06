import { Logo } from "./logo";

export function SiteHeader() {
  return (
    <header className="flex items-center justify-between px-5 py-5 sm:px-8 sm:py-6">
      <Logo />
      <p className="text-sm text-neutral-500">Virtual Try-On</p>
    </header>
  );
}
