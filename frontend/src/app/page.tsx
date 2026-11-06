import { SiteHeader } from "@/components/site-header";
import { TryOnExperience } from "@/components/try-on-experience";

export default function Home() {
  return (
    <div className="flex flex-1 justify-center p-3 sm:p-6 lg:p-10">
      <div className="flex w-full max-w-5xl flex-col rounded-3xl border border-neutral-200 bg-white">
        <SiteHeader />
        <main className="flex flex-col gap-12 px-5 pt-6 pb-10 sm:px-10 sm:pt-10 lg:px-16 lg:pb-16">
          <TryOnExperience />
        </main>
      </div>
    </div>
  );
}
