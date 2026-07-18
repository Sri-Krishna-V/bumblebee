import { BrandMark } from "@/components/brand-mark";
import { Workbench } from "@/components/workbench";

const proofPoints = [
  ["01", "Source-grounded", "Keep the page, region, and chunk relationship inspectable."],
  ["02", "RAG-native", "Deliver Markdown plus retrieval-ready chunks in one response."],
  ["03", "Pilot-safe", "No document archive; only a short, metadata-only audit trail."],
];

export default function Home() {
  return (
    <main>
      <header className="site-header">
        <a className="brand" href="#top" aria-label="Bumblebee home">
          <BrandMark />
          <span>Bumblebee <small>evidence engine</small></span>
        </a>
        <nav aria-label="Page sections">
          <a href="#studio">Studio</a>
          <a href="#proof">Why it holds</a>
          <a className="nav-cta" href="#studio">Start a trace <span>↗</span></a>
        </nav>
      </header>

      <section className="hero" id="top">
        <div className="hero-copy">
          <p className="eyebrow">Private ingestion for evidence-heavy teams</p>
          <h1>Make the <span>PDF</span><br />answerable.</h1>
          <p className="hero-deck">Bumblebee turns unruly documents into source-grounded Markdown and durable retrieval chunks—without treating your archive like training data.</p>
          <div className="hero-actions">
            <a className="button button--ink" href="#studio">Open the studio <span>↓</span></a>
            <a className="text-link" href="#proof">See the operating promise <span>→</span></a>
          </div>
        </div>
        <div className="hero-art" aria-label="An illustration of PDF pages becoming inspectable evidence">
          <div className="pollen pollen-one" />
          <div className="pollen pollen-two" />
          <div className="path-line" />
          <article className="paper-card paper-card--source">
            <span className="card-kicker">SOURCE / p. 08</span>
            <h2>Appendix B<br />Material Terms</h2>
            <i /><i /><i /><i />
            <mark>Risk allocation</mark>
          </article>
          <article className="paper-card paper-card--evidence">
            <span className="card-kicker">EVIDENCE / chunk 014</span>
            <strong>Risk allocation</strong>
            <p>Supplier shall notify the buyer within five business days…</p>
            <footer><b>0.94</b> region confidence <span>✓</span></footer>
          </article>
          <div className="hex-orbit"><BrandMark /></div>
          <p className="art-caption">Document → regions → evidence</p>
        </div>
      </section>

      <section className="signal-band" aria-label="Bumblebee product signals">
        <p><span>BUILT FOR</span> research ops · legal intelligence · policy teams · technical archives</p>
        <p><span>OUTPUT</span> Markdown · chunks · layout context · trace receipt</p>
      </section>

      <Workbench />

      <section className="proof" id="proof">
        <div className="proof-heading">
          <p className="eyebrow">The useful moat is trust</p>
          <h2>Speed gets a first look.<br /><em>Evidence earns the workflow.</em></h2>
        </div>
        <div className="proof-grid">
          {proofPoints.map(([number, title, copy]) => (
            <article key={number}>
              <span>{number}</span>
              <h3>{title}</h3>
              <p>{copy}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="closing-callout">
        <BrandMark />
        <div><p className="eyebrow">Design partner program</p><h2>Bring one difficult document set.</h2></div>
        <a className="button button--amber" href="#studio">Trace it with us <span>↗</span></a>
      </section>

      <footer className="site-footer">
        <p>© {new Date().getFullYear()} Bumblebee. Built for documents that need to hold up.</p>
        <p>PDFs are not retained by the pilot API.</p>
      </footer>
    </main>
  );
}
