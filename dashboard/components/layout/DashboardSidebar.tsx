"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  ClipboardList,
  LineChart,
  LogOut,
  Pill,
  Package,
  Tags,
  Upload,
} from "lucide-react";

import { useAuth } from "@/contexts/AuthContext";

const NAV_ITEMS = [
  {
    href: "/data-ingestion",
    label: "Data Ingestion",
    description: "Upload receipt spreadsheets",
    icon: Upload,
  },
  {
    href: "/drugs",
    label: "Drug Registry",
    description: "Unique drugs from receipts",
    icon: ClipboardList,
  },
  {
    href: "/categories",
    label: "Category Registry",
    description: "Unique categories from receipts",
    icon: Tags,
  },
  {
    href: "/cold-start",
    label: "Cold Start",
    description: "New drug predictions",
    icon: Pill,
  },
  {
    href: "/forecasting",
    label: "Forecasting",
    description: "Ensemble demand models",
    icon: LineChart,
  },
] as const;

export function DashboardSidebar() {
  const pathname = usePathname();
  const { user, logout } = useAuth();

  return (
    <aside className="flex w-64 shrink-0 flex-col border-r border-slate-200 bg-white">
      <div className="border-b border-slate-100 px-5 py-6">
        <div className="flex items-center gap-3">
          <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-blue-600 text-white shadow-sm">
            <Package className="h-5 w-5" />
          </span>
          <div>
            <p className="text-sm font-bold tracking-tight text-slate-900">Supply Chain</p>
            <p className="text-xs text-slate-500">Demand forecasting</p>
          </div>
        </div>
      </div>

      <nav className="flex-1 space-y-1 p-3" aria-label="Dashboard modules">
        {NAV_ITEMS.map((item) => {
          const active = pathname === item.href || pathname.startsWith(`${item.href}/`);
          const Icon = item.icon;

          return (
            <Link
              key={item.href}
              href={item.href}
              className={`flex items-start gap-3 rounded-xl px-3 py-3 transition ${
                active
                  ? "bg-blue-50 ring-1 ring-blue-200"
                  : "text-slate-600 hover:bg-slate-50 hover:text-slate-900"
              }`}
            >
              <span
                className={`mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg ${
                  active ? "bg-blue-600 text-white" : "bg-slate-100 text-slate-500"
                }`}
              >
                <Icon className="h-4 w-4" />
              </span>
              <span>
                <span
                  className={`block text-sm font-semibold ${
                    active ? "text-blue-900" : "text-slate-800"
                  }`}
                >
                  {item.label}
                </span>
                <span className="block text-xs text-slate-500">{item.description}</span>
              </span>
            </Link>
          );
        })}
      </nav>

      <div className="border-t border-slate-100 px-5 py-4">
        {user ? (
          <p className="mb-3 truncate text-xs font-medium text-slate-600" title={user.email}>
            {user.email}
          </p>
        ) : null}
        <button
          type="button"
          onClick={logout}
          className="flex w-full items-center gap-2 rounded-lg px-2 py-2 text-sm font-medium text-slate-600 transition hover:bg-slate-50 hover:text-slate-900"
        >
          <LogOut className="h-4 w-4" />
          Sign out
        </button>
        <p className="mt-3 text-[11px] leading-relaxed text-slate-400">
          Validation dashboard for hospital drug supply chain forecasting modules.
        </p>
      </div>
    </aside>
  );
}
