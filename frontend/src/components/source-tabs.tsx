import { useId, useRef, type KeyboardEvent } from "react";
import { ImageIcon, Link2 } from "lucide-react";
import { ProductLinkForm } from "./product-link-form";
import { ImageUpload } from "./image-upload";
import type { ReadyGlasses } from "@/lib/api";

const TABS = [
  { key: "link", label: "Product link", icon: Link2 },
  { key: "image", label: "Image upload", icon: ImageIcon },
] as const;

export type TabKey = (typeof TABS)[number]["key"];

/** WAI-ARIA tabs: arrow keys, Home and End move between tabs. Both panels stay mounted to keep their state. */
export function SourceTabs({
  active,
  onActiveChange: setActive,
  onReady,
}: {
  active: TabKey;
  onActiveChange: (tab: TabKey) => void;
  onReady: (glasses: ReadyGlasses) => void;
}) {
  const baseId = useId();
  const tabRefs = useRef<Record<TabKey, HTMLButtonElement | null>>({ link: null, image: null });

  function handleKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    const index = TABS.findIndex((tab) => tab.key === active);
    const next = {
      ArrowRight: (index + 1) % TABS.length,
      ArrowLeft: (index - 1 + TABS.length) % TABS.length,
      Home: 0,
      End: TABS.length - 1,
    }[event.key];
    if (next === undefined) return;
    event.preventDefault();
    const key = TABS[next].key;
    setActive(key);
    tabRefs.current[key]?.focus();
  }

  return (
    <div className="flex flex-col gap-5">
      <div
        role="tablist"
        aria-label="How to find your glasses"
        onKeyDown={handleKeyDown}
        className="inline-flex self-center rounded-xl bg-neutral-100 p-1"
      >
        {TABS.map(({ key, label, icon: Icon }) => {
          const selected = key === active;
          return (
            <button
              key={key}
              ref={(element) => {
                tabRefs.current[key] = element;
              }}
              type="button"
              role="tab"
              id={`${baseId}-tab-${key}`}
              aria-selected={selected}
              aria-controls={`${baseId}-panel-${key}`}
              tabIndex={selected ? 0 : -1}
              onClick={() => setActive(key)}
              className={`inline-flex h-10 items-center gap-2 rounded-lg px-4 text-sm font-medium transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-neutral-950 sm:px-5 ${
                selected ? "bg-white text-neutral-950 shadow-xs" : "text-neutral-500 hover:text-neutral-800"
              }`}
            >
              <Icon aria-hidden className="size-4" />
              {label}
            </button>
          );
        })}
      </div>

      {TABS.map(({ key }) => (
        <div
          key={key}
          role="tabpanel"
          id={`${baseId}-panel-${key}`}
          aria-labelledby={`${baseId}-tab-${key}`}
          hidden={key !== active}
        >
          {key === "link" ? (
            <ProductLinkForm
              onReady={onReady}
              onUseUpload={() => {
                setActive("image");
                tabRefs.current.image?.focus();
              }}
            />
          ) : (
            <ImageUpload onReady={onReady} />
          )}
        </div>
      ))}
    </div>
  );
}
