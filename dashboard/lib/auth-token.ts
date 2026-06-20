const STORAGE_KEY = "auth_access_token";
const COOKIE_NAME = "auth_token";

function maxAgeSeconds(): number {
  return 60 * 60 * 24;
}

export function getAccessToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem(STORAGE_KEY);
}

export function setAccessToken(token: string): void {
  localStorage.setItem(STORAGE_KEY, token);
  const maxAge = maxAgeSeconds();
  document.cookie = `${COOKIE_NAME}=${encodeURIComponent(token)}; path=/; max-age=${maxAge}; SameSite=Lax`;
}

export function clearAccessToken(): void {
  localStorage.removeItem(STORAGE_KEY);
  document.cookie = `${COOKIE_NAME}=; path=/; max-age=0; SameSite=Lax`;
}

export function authHeaders(): HeadersInit {
  const token = getAccessToken();
  if (!token) return {};
  return { Authorization: `Bearer ${token}` };
}
