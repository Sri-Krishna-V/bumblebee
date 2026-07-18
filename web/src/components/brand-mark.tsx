import type { SVGProps } from "react";

export function BrandMark({ className, ...props }: SVGProps<SVGSVGElement>) {
  return (
    <svg
      aria-hidden="true"
      className={className}
      fill="none"
      viewBox="0 0 52 52"
      xmlns="http://www.w3.org/2000/svg"
      {...props}
    >
      <path d="m26 3 19.9 11.5v23L26 49 6.1 37.5v-23L26 3Z" fill="currentColor" />
      <path d="M26 12.5 37.1 19v12.8L26 38.3l-11.1-6.5V19L26 12.5Z" fill="#F7F0E4" />
      <path d="M21 23.4c2.6-2.2 7.4-2.2 10 0M20 29.1c3.2 2.2 8.8 2.2 12 0" stroke="currentColor" strokeWidth="2.7" />
    </svg>
  );
}
