"use client";

export default function ErrorPage({ reset }: { reset: () => void }) {
  return (
    <main className="error-page">
      <p className="eyebrow">The hive lost the thread</p>
      <h1>That document path did not hold.</h1>
      <p>Nothing was saved. You can try the request again.</p>
      <button className="button button--ink" onClick={reset} type="button">
        Try again
      </button>
    </main>
  );
}
