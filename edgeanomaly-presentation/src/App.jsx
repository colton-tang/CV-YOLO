import { useEffect, useRef, useState } from 'react'
import {
  ArrowRight,
  Camera,
  ChevronLeft,
  ChevronRight,
  Clock3,
  Cpu,
  Download,
  FileImage,
  GitBranch,
  Layers3,
  ListTree,
  MonitorPlay,
  Radar,
  ShieldCheck,
  SquareDashedMousePointer,
} from 'lucide-react'

const STATUS_STYLES = {
  RESOLVED: 'bg-emerald-50 text-emerald-700 border-emerald-200',
  VERIFYING: 'bg-amber-50 text-amber-700 border-amber-200',
  OUTLIER: 'bg-rose-50 text-rose-700 border-rose-200',
  UNKNOWN: 'bg-sky-50 text-sky-700 border-sky-200',
}

const slides = [
  {
    id: 'cover',
    type: 'cover',
    kicker: 'OOD Detection Framework Walkthrough',
    title: 'EdgeAnomalyCCTV',
    subtitle: 'An edge pipeline that fast-paths known objects and escalates OOD candidates for deeper verification.',
    valueHook:
      'The key design is selective OOD handling: high-confidence inliers pass immediately, while uncertain or unknown objects are isolated, reviewed, and labeled through an asynchronous vision-language path.',
    author: {
      project: 'Derived from EdgeAnomalyCCTV/src/main.py',
      focus: 'Framework structure, OOD handling, and runtime flow',
    },
  },
  {
    id: 'architecture',
    type: 'architecture',
    header: '01. Five-Layer Architecture',
    tagline: 'System Map',
    layers: [
      {
        title: 'Layer 1',
        label: 'Ingestion',
        name: 'IngestionLayer',
        icon: <Camera size={22} />,
        points: ['Normalizes image, file, URL, or camera input', 'Delivers unified frame data for downstream OOD decisions'],
      },
      {
        title: 'Layer 2',
        label: 'Detection and Tracking',
        name: 'DetectionTrackingLayer',
        icon: <Radar size={22} />,
        points: ['Runs YOLO detection and BotSort tracking', 'Produces class, confidence, bbox, and track history used to spot OOD candidates'],
      },
      {
        title: 'Layer 3',
        label: 'OOD Gate Filter',
        name: 'GateOutlierFilterLayer',
        icon: <GitBranch size={22} />,
        points: ['Separates inliers from uncertain or unknown objects', 'Stores per-track OOD state in track_state_db'],
      },
      {
        title: 'Layer 4',
        label: 'OOD Verifier',
        name: 'LLMClassifierLayer',
        icon: <Cpu size={22} />,
        points: ['Consumes llm_queue asynchronously', 'Uses Qwen3-VL-2B-Instruct to confirm known vs OOD crops'],
      },
      {
        title: 'Layer 5',
        label: 'Render and Alert',
        name: 'RenderAlertLayer',
        icon: <MonitorPlay size={22} />,
        points: ['Renders operator-visible OOD outcomes', 'Prints summaries and status-aware annotations for alerts'],
      },
    ],
  },
  {
    id: 'orchestration',
    type: 'orchestration',
    header: '02. main.py Orchestration',
    tagline: 'Runtime Control',
    stages: [
      'Parse --mode and --input, or fall back to interactive mode selection.',
      'Resolve the default input: benchmark image for graph mode, camera 0 for video mode.',
      'Instantiate all five layers and start classifier.run() as a background asyncio task.',
      'Branch into single-frame graph execution or continuous video loop while preserving the OOD review path.',
    ],
    branches: [
      {
        title: 'Graph mode OOD path',
        icon: <FileImage size={20} />,
        items: ['Single call to ingestion.get_frame()', 'Wait for llm_queue.join() so every OOD candidate is resolved', 'Render final annotated still image once'],
      },
      {
        title: 'Video mode OOD path',
        icon: <MonitorPlay size={20} />,
        items: ['OpenCV capture with camera/file/URL support', 'Per-frame detect -> OOD gate -> render loop with async review in parallel', "Exit on 'q' or closed render window"],
      },
    ],
  },
  {
    id: 'gates',
    type: 'gates',
    header: '03. OOD Decision Flow',
    tagline: 'State Machine',
    statuses: ['RESOLVED', 'VERIFYING', 'OUTLIER', 'UNKNOWN'],
    example: {
      image: '/generated/bus-yolov8n-output.jpg',
      title: 'Example detector run',
      subtitle: 'main.py in graph mode using base yolov8n on benchmark_data/legacy/bus.jpg',
      facts: ['Model: yolov8n', 'Objects found: 1 bus, 3 people', 'Outcome: all fast-pathed to RESOLVED'],
    },
    gates: [
      {
        title: 'Gate 1: Finalized?',
        body: 'If a track is already RESOLVED, OUTLIER, or UNKNOWN, it is dropped immediately so OOD work is not repeated.',
      },
      {
        title: 'Gate 2: Fast-path known object',
        body: 'If confidence exceeds HIGH_CONFIDENCE_THRESHOLD and the class is known, the object is treated as in-distribution and auto-passes to RESOLVED.',
      },
      {
        title: 'Gate 3: OOD candidate',
        body: 'Tracks in the uncertainty band or outside known_classes are treated as potential OOD objects, enter VERIFYING, and are queued for vision-language review once.',
      },
      {
        title: 'Below low threshold',
        body: 'Detections below LOW_CONFIDENCE_THRESHOLD are ignored so the OOD path is reserved for meaningful candidates rather than noise.',
      },
    ],
  },
  {
    id: 'async',
    type: 'async',
    header: '04. Smart Review for Unusual Objects',
    tagline: 'Background Verification',
    flow: [
      { name: 'Scan the scene', detail: 'The system checks each frame and identifies what is happening.' },
      { name: 'Pass routine objects', detail: 'Clearly recognized objects move through immediately with no extra delay.' },
      { name: 'Escalate unusual cases', detail: 'Only unclear or suspicious objects are sent for deeper review.' },
      { name: 'Update the operator', detail: 'The result appears as soon as the background review finishes.' },
    ],
    summary:
      'The system does not pause the whole video feed for every object. It keeps normal monitoring moving and spends extra attention only where risk is higher.',
    benefits: [
      'Faster day-to-day monitoring',
      'Deeper review only when needed',
      'A better balance of speed and accuracy',
    ],
  },
  {
    id: 'benchmark',
    type: 'benchmark',
    header: '05. Performance Benchmarks',
    tagline: 'Matrix Evaluation',
    variants: [
      {
        name: 'yolov8n_only',
        oodDetectionRate: '0.00%',
        llmJudgeAccuracy: '0.00%',
      },
      {
        name: 'yolov8n_framework',
        oodDetectionRate: '0.00%',
        llmJudgeAccuracy: '12.07%',
      },
      {
        name: 'yolo_world_only',
        oodDetectionRate: '76.92%',
        llmJudgeAccuracy: '5.13%',
      },
      {
        name: 'yolo_world_framework',
        oodDetectionRate: '76.92%',
        llmJudgeAccuracy: '16.67%',
      },
      {
        name: 'vlm_only',
        oodDetectionRate: '—',
        llmJudgeAccuracy: '—',
      },
    ],
  },
  {
    id: 'future',
    type: 'future',
    header: '06. Future Development',
    tagline: 'Roadmap & Horizon',
    objectives: [
      {
        title: 'YOLO Fine-Tuning for OOD',
        icon: <Radar size={22} />,
        detail: "Fine-tune the YOLO detector on target-domain inliers. This sharpens boundary features and increases the model's class-specific confidence gap, improving the accuracy of initial OOD gate filters.",
      },
      {
        title: 'VLM Distillation to the Edge',
        icon: <Cpu size={22} />,
        detail: 'Distill heavy multimodal reasoning (e.g., Qwen-VL) into a compact, specialized edge classifier, or leverage quantized INT4/INT8 formats to fit tight VRAM constraints.',
      },
      {
        title: 'Active Learning Feedback Loop',
        icon: <GitBranch size={22} />,
        detail: 'Establish an automatic pipeline where edge-detected OOD outliers are flagged, reviewed by human operators, and sent back to a central server to continuously retrain the detectors.',
      },
      {
        title: 'TensorRT & ONNX Acceleration',
        icon: <Clock3 size={22} />,
        detail: 'Port the YOLO detector and the local verification logic to TensorRT or ONNX Runtime to minimize latency and maximize frame throughput on target edge devices.',
      },
    ],
  },
]

function App() {
  const [currentSlide, setCurrentSlide] = useState(0)
  const [isExporting, setIsExporting] = useState(false)
  const exportSlideRefs = useRef([])

  const next = () => setCurrentSlide((prev) => (prev === slides.length - 1 ? prev : prev + 1))
  const prev = () => setCurrentSlide((prev) => (prev === 0 ? prev : prev - 1))

  const setExportSlideRef = (index) => (node) => {
    exportSlideRefs.current[index] = node
  }

  const getSearchableTextItems = (slideElement) => {
    const slideRect = slideElement.getBoundingClientRect()
    const walker = document.createTreeWalker(slideElement, NodeFilter.SHOW_TEXT)
    const textItems = []

    while (walker.nextNode()) {
      const textNode = walker.currentNode
      const text = textNode.textContent || ''
      const parent = textNode.parentElement

      if (!parent || !text.trim()) continue

      const style = window.getComputedStyle(parent)
      if (style.visibility === 'hidden' || style.display === 'none' || Number(style.opacity) === 0) {
        continue
      }

      const matches = text.matchAll(/\S+/g)
      for (const match of matches) {
        const word = match[0]
        const start = match.index
        const end = start + word.length
        const range = document.createRange()
        range.setStart(textNode, start)
        range.setEnd(textNode, end)

        const rect = Array.from(range.getClientRects()).find((item) => item.width > 0 && item.height > 0)
        range.detach()

        if (!rect) continue

        textItems.push({
          text: word,
          x: rect.left - slideRect.left,
          y: rect.top - slideRect.top,
          width: rect.width,
          height: rect.height,
        })
      }
    }

    return textItems
  }

  const exportSlidesToPdf = async () => {
    if (isExporting || !exportSlideRefs.current.length) return
    setIsExporting(true)

    try {
      const [{ default: html2canvas }, { jsPDF }] = await Promise.all([import('html2canvas'), import('jspdf')])
      const exportWidth = 1600
      const exportHeight = 900
      const pageImages = []

      if (document.fonts?.ready) {
        await document.fonts.ready
      }

      for (let slideIndex = 0; slideIndex < exportSlideRefs.current.length; slideIndex += 1) {
        const slideElement = exportSlideRefs.current[slideIndex]
        if (!slideElement) continue

        const searchableTextItems = getSearchableTextItems(slideElement)
        const canvas = await html2canvas(slideElement, {
          backgroundColor: '#f8fafc',
          scale: 2,
          useCORS: true,
          scrollX: 0,
          scrollY: 0,
          width: exportWidth,
          height: exportHeight,
          windowWidth: exportWidth,
          windowHeight: exportHeight,
        })

        pageImages.push({
          dataUrl: canvas.toDataURL('image/jpeg', 0.95),
          searchableTextItems,
        })
      }

      if (!pageImages.length) return

      const pdf = new jsPDF({
        orientation: 'landscape',
        unit: 'pt',
        format: [1600, 900],
        compress: true,
      })

      pageImages.forEach((page, index) => {
        if (index > 0) {
          pdf.addPage([1600, 900], 'landscape')
        }

        pdf.setFont('helvetica', 'normal')
        pdf.setTextColor(0, 0, 0)

        page.searchableTextItems.forEach((item) => {
          const fontSize = Math.max(4, item.height * 0.92)
          const characterCount = Array.from(item.text).length
          pdf.setFontSize(fontSize)
          const pdfTextWidth = pdf.getTextWidth(item.text)
          const charSpace = characterCount > 1 ? (item.width - pdfTextWidth) / (characterCount - 1) : 0
          pdf.text(item.text, item.x, item.y, { baseline: 'top', charSpace })
        })

        pdf.addImage(page.dataUrl, 'JPEG', 0, 0, 1600, 900, undefined, 'SLOW')
      })

      pdf.save('edgeanomalycctv-framework.pdf')
    } finally {
      setIsExporting(false)
    }
  }

  useEffect(() => {
    const handleKeyDown = (event) => {
      if (event.key === 'ArrowRight' || event.key === ' ') next()
      if (event.key === 'ArrowLeft') prev()
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [])

  const slide = slides[currentSlide]

  return (
    <div className="deck-shell text-slate-900">
      <div className="mx-auto flex min-h-screen max-w-[1800px] flex-col px-4 py-6 md:px-8">
        <header className="mb-5 flex items-center justify-between gap-4 rounded-[22px] border border-slate-200/70 bg-[rgba(255,252,246,0.88)] px-5 py-4 shadow-panel backdrop-blur">
          <div>
            <p className="section-label">Technical Deck</p>
            <h1 className="mt-1 text-lg font-semibold text-slate-900">EdgeAnomalyCCTV OOD Framework</h1>
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={exportSlidesToPdf}
              disabled={isExporting}
              className="inline-flex items-center gap-2 rounded-[14px] border border-slate-300 bg-white px-4 py-2 text-sm font-semibold text-slate-700 transition hover:border-slate-400 hover:bg-slate-50 disabled:cursor-wait disabled:opacity-70"
            >
              <Download size={16} />
              {isExporting ? 'Exporting...' : 'Export PDF'}
            </button>
            <div className="rounded-[14px] border border-slate-200 bg-white px-4 py-2 text-sm font-semibold text-slate-600">
              {currentSlide + 1} / {slides.length}
            </div>
          </div>
        </header>

        <main className="flex flex-1 flex-col justify-center">
          <SlideView slide={slide} slideIndex={currentSlide} />
        </main>

        <footer className="mt-5 flex flex-wrap items-center justify-between gap-3">
          <p className="text-sm text-slate-500">
            Use <span className="font-semibold text-slate-700">ArrowLeft</span>,{' '}
            <span className="font-semibold text-slate-700">ArrowRight</span>, or space to navigate.
          </p>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={prev}
              disabled={currentSlide === 0}
              className="inline-flex items-center gap-2 rounded-[14px] border border-slate-300 bg-[rgba(255,252,246,0.9)] px-4 py-2 text-sm font-semibold text-slate-700 shadow-sm transition hover:border-slate-400 disabled:cursor-not-allowed disabled:opacity-50"
            >
              <ChevronLeft size={16} />
              Previous
            </button>
            <button
              type="button"
              onClick={next}
              disabled={currentSlide === slides.length - 1}
              className="inline-flex items-center gap-2 rounded-[14px] bg-slate-900 px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-50"
            >
              Next
              <ChevronRight size={16} />
            </button>
          </div>
        </footer>
      </div>

      <div className="pointer-events-none fixed -left-[99999px] top-0">
        {slides.map((item, index) => (
          <SlideView
            key={item.id}
            slide={item}
            slideIndex={index}
            mode="export"
            slideRef={setExportSlideRef(index)}
          />
        ))}
      </div>
    </div>
  )
}

function SlideView({ slide, slideIndex, mode = 'screen', slideRef }) {
  const isExportMode = mode === 'export'
  const shellClass = isExportMode
    ? 'slide-panel diagram-grid flex h-[900px] w-[1600px] flex-col overflow-hidden rounded-none p-16'
    : 'slide-panel diagram-grid mx-auto aspect-[16/9] w-full max-w-7xl rounded-[28px] p-6 shadow-panel backdrop-blur md:p-10'

  return (
    <section ref={slideRef} className={shellClass}>
      {slide.type === 'cover' ? (
        <CoverSlide slide={slide} />
      ) : (
        <StandardSlide slide={slide} slideIndex={slideIndex} isExportMode={isExportMode} />
      )}
    </section>
  )
}

function CoverSlide({ slide }) {
  return (
    <div className="flex h-full flex-col justify-between">
      <div className="grid gap-6 md:grid-cols-[1.38fr_0.62fr]">
        <div className="max-w-5xl space-y-5">
          <span className="section-label">{slide.kicker}</span>
          <div className="space-y-4">
            <h2 className="display-serif max-w-4xl text-[4.4rem] leading-[0.9] text-slate-900 md:text-[5rem]">
              {slide.title}
            </h2>
            <p className="max-w-4xl text-[1.3rem] leading-relaxed text-slate-600 md:text-[1.55rem]">{slide.subtitle}</p>
          </div>
          <div className="deck-card max-w-4xl rounded-[22px] p-6">
            <p className="section-label text-emerald-700">Core Thesis</p>
            <p className="mt-3 text-[1.08rem] leading-relaxed text-slate-700 md:text-[1.28rem]">{slide.valueHook}</p>
          </div>
        </div>

        <div className="deck-card-dark rounded-[24px] p-6 text-white">
          <p className="section-label text-cyan-200">OOD Strategy</p>
          <div className="mt-5 space-y-3 text-sm leading-relaxed text-slate-200">
            <div className="subtle-rule pt-3">
              <p className="text-xs uppercase tracking-[0.18em] text-slate-400">Fast Path</p>
              <p className="mt-2 text-base text-white">Known + high confidence {'->'} <span className="mono-note">RESOLVED</span></p>
            </div>
            <div className="subtle-rule pt-3">
              <p className="text-xs uppercase tracking-[0.18em] text-slate-400">Review Path</p>
              <p className="mt-2 text-base text-white">Uncertain or unknown {'->'} <span className="mono-note">VERIFYING</span> {'->'} VLM decision</p>
            </div>
            <div className="subtle-rule pt-3">
              <p className="text-xs uppercase tracking-[0.18em] text-slate-400">Operational Goal</p>
              <p className="mt-2 text-base text-white">Keep video throughput fast while escalating suspicious objects selectively.</p>
            </div>
          </div>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-[1.25fr_0.75fr]">
        <div className="deck-card-dark rounded-[22px] p-6 text-white">
          <p className="section-label text-cyan-200">Pipeline Shape</p>
          <div className="mt-4 flex flex-wrap items-center gap-3 text-sm font-semibold">
            {['Ingestion', 'Detection', 'OOD Gate', 'VLM Review', 'Render'].map((item, index) => (
              <div key={item} className="flex items-center gap-3">
                <span className="rounded-[12px] border border-white/10 bg-white/5 px-4 py-2">{item}</span>
                {index < 4 && <ArrowRight size={16} className="text-cyan-200" />}
              </div>
            ))}
          </div>
        </div>
        <div className="deck-card rounded-[22px] p-6">
          <p className="section-label text-slate-500">Presentation Focus</p>
          <div className="mt-4 space-y-3 text-base text-slate-700">
            <p>{slide.author.project}</p>
            <p>{slide.author.focus}</p>
          </div>
        </div>
      </div>
    </div>
  )
}

function StandardSlide({ slide, slideIndex, isExportMode }) {
  return (
    <div className="flex h-full flex-col">
      <div className="mb-8 flex items-start justify-between gap-4">
        <div className="max-w-4xl">
          <p className="section-label mb-3">{slide.tagline}</p>
          <h2 className={`${isExportMode ? 'text-[3.6rem]' : 'text-[2.65rem] md:text-[4rem]'} display-serif leading-[0.93] text-slate-900`}>
            {slide.header}
          </h2>
        </div>
        <div className="rounded-[18px] border border-slate-200 bg-[rgba(255,252,246,0.85)] px-5 py-3 text-4xl font-black text-slate-200">
          0{slideIndex}
        </div>
      </div>

      <div className="flex-1">
        {slide.type === 'architecture' && <ArchitectureSlide slide={slide} />}
        {slide.type === 'orchestration' && <OrchestrationSlide slide={slide} />}
        {slide.type === 'gates' && <GatesSlide slide={slide} />}
        {slide.type === 'async' && <AsyncSlide slide={slide} />}
        {slide.type === 'benchmark' && <BenchmarkSlide slide={slide} />}
        {slide.type === 'future' && <FutureSlide slide={slide} />}
      </div>
    </div>
  )
}

function ArchitectureSlide({ slide }) {
  return (
    <div className="grid h-full gap-5 md:grid-cols-5">
      {slide.layers.map((layer) => (
        <div key={layer.name} className="deck-card flex flex-col rounded-[22px] p-5">
          <div className="mb-4 flex items-center justify-between">
            <span className="rounded-[10px] bg-slate-100 px-3 py-1 text-xs font-bold uppercase tracking-[0.18em] text-slate-500">
              {layer.title}
            </span>
            <div className="rounded-[12px] bg-cyan-50 p-3 text-cyan-700">{layer.icon}</div>
          </div>
          <div className="min-h-[5.9rem]">
            <h3 className="text-[1.08rem] font-semibold leading-snug text-slate-900 md:text-[1.2rem]">
              {layer.label}
            </h3>
            <p className="mono-note mt-2 break-words text-[0.76rem] leading-relaxed text-cyan-800 md:text-[0.8rem]">
              {layer.name}
            </p>
          </div>
          <div className="mt-4 space-y-3 text-sm leading-relaxed text-slate-600">
            {layer.points.map((point) => (
              <p key={point}>{point}</p>
            ))}
          </div>
        </div>
      ))}
    </div>
  )
}

function OrchestrationSlide({ slide }) {
  return (
    <div className="grid h-full gap-6 md:grid-cols-[1.15fr_0.85fr]">
      <div className="deck-card rounded-[22px] p-6">
        <div className="mb-5 flex items-center gap-3">
          <Layers3 className="text-cyan-700" />
          <h3 className="text-2xl font-semibold text-slate-900">Startup and control path</h3>
        </div>
        <div className="space-y-4">
          {slide.stages.map((stage, index) => (
            <div key={stage} className="flex gap-4 rounded-[18px] border border-slate-200/80 bg-slate-50/85 p-4">
              <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-[12px] bg-slate-900 text-sm font-bold text-white">
                {index + 1}
              </div>
              <p className="text-base leading-relaxed text-slate-700">{stage}</p>
            </div>
          ))}
        </div>
      </div>

      <div className="space-y-5">
        {slide.branches.map((branch) => (
          <div key={branch.title} className="deck-card-dark rounded-[22px] p-6 text-white">
            <div className="mb-4 flex items-center gap-3">
              <div className="rounded-[12px] bg-white/10 p-3">{branch.icon}</div>
              <h3 className="text-2xl font-semibold">{branch.title}</h3>
            </div>
            <div className="space-y-3 text-sm leading-relaxed text-slate-200">
              {branch.items.map((item) => (
                <p key={item}>{item}</p>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

function GatesSlide({ slide }) {
  return (
    <div className="grid h-full gap-5 md:grid-cols-[0.74fr_1.26fr]">
      <div className="space-y-4">
        <div className="deck-card rounded-[22px] p-5">
          <div className="mb-4 flex items-center gap-3">
            <ShieldCheck className="text-cyan-700" />
            <h3 className="text-2xl font-semibold text-slate-900">State outcomes</h3>
          </div>
          <div className="grid gap-3">
            {slide.statuses.map((status) => (
              <div key={status} className={`rounded-[16px] border px-4 py-3 text-base font-semibold ${STATUS_STYLES[status]}`}>
                {status}
              </div>
            ))}
          </div>
        </div>

        <div className="deck-card-dark rounded-[22px] p-5 text-white">
          <p className="section-label text-cyan-200">Real Example</p>
          <h3 className="mt-3 text-[1.35rem] font-semibold leading-snug">{slide.example.title}</h3>
          <p className="mt-2 text-sm leading-relaxed text-slate-200">{slide.example.subtitle}</p>
          <div className="mt-4 space-y-2 text-sm leading-relaxed text-slate-100">
            {slide.example.facts.map((fact) => (
              <p key={fact}>{fact}</p>
            ))}
          </div>
        </div>
      </div>

      <div className="grid gap-4">
        <div className="deck-card overflow-hidden rounded-[22px] p-4">
          <div className="mb-3 flex items-center justify-between gap-4">
            <div>
              <p className="section-label">Annotated Output</p>
              <h3 className="mt-2 text-xl font-semibold text-slate-900">Base `yolov8n` result on `bus.jpg`</h3>
            </div>
            <span className="rounded-[12px] border border-emerald-200 bg-emerald-50 px-3 py-2 text-sm font-semibold text-emerald-700">
              All detections resolved
            </span>
          </div>
          <div className="overflow-hidden rounded-[18px] border border-slate-200 bg-slate-100">
            <img
              src={slide.example.image}
              alt="Annotated detector output showing a bus and three people labeled as resolved."
              className="h-[21.5rem] w-full object-cover object-center"
            />
          </div>
        </div>

        <div className="grid gap-3 md:grid-cols-2">
          {slide.gates.map((gate) => (
            <div key={gate.title} className="deck-card rounded-[20px] p-4">
              <div className="mb-2 flex items-center gap-3">
                <SquareDashedMousePointer className="text-amber-600" size={18} />
                <h3 className="text-lg font-semibold text-slate-900 md:text-xl">{gate.title}</h3>
              </div>
              <p className="text-[0.98rem] leading-relaxed text-slate-600">{gate.body}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

function AsyncSlide({ slide }) {
  return (
    <div className="grid h-full gap-5 md:grid-cols-[1.02fr_0.98fr]">
      <div className="deck-card rounded-[22px] p-6">
        <div className="mb-4 flex items-center gap-3">
          <ListTree className="text-cyan-700" />
          <h3 className="text-2xl font-semibold text-slate-900">How the review works</h3>
        </div>
        <div className="space-y-3">
          {slide.flow.map((step, index) => (
            <div key={step.name} className="flex items-start gap-3 rounded-[18px] border border-slate-200/80 bg-slate-50/80 p-4">
              <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-[12px] bg-cyan-700 text-sm font-bold text-white">
                {index + 1}
              </div>
              <div className="flex-1">
                <h4 className="text-[1rem] font-semibold text-slate-900 md:text-[1.08rem]">{step.name}</h4>
                <p className="mt-1 text-[0.95rem] leading-relaxed text-slate-600">{step.detail}</p>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="grid gap-4 md:grid-rows-[auto_auto_1fr]">
        <div className="deck-card rounded-[22px] border-cyan-200 bg-cyan-50/80 p-6">
          <div className="mb-3 flex items-center gap-3 text-cyan-700">
            <Cpu size={22} />
            <h3 className="text-xl font-semibold">Why this matters</h3>
          </div>
          <p className="text-base leading-relaxed text-cyan-900">{slide.summary}</p>
        </div>

        <div className="deck-card rounded-[22px] p-6">
          <p className="section-label">Stakeholder Value</p>
          <div className="mt-4 space-y-3">
            {slide.benefits.map((benefit) => (
              <div key={benefit} className="rounded-[16px] border border-slate-200 bg-slate-50/80 px-4 py-3 text-[0.98rem] font-medium text-slate-700">
                {benefit}
              </div>
            ))}
          </div>
        </div>

        <div className="deck-card-dark rounded-[22px] p-6 text-base leading-relaxed text-slate-100">
          Unusual objects still get extra scrutiny, but the rest of the scene keeps moving without interruption.
        </div>
      </div>
    </div>
  )
}

function FutureSlide({ slide }) {
  return (
    <div className="grid h-full gap-5 md:grid-cols-2 md:grid-rows-2">
      {slide.objectives.map((obj) => (
        <div key={obj.title} className="deck-card flex flex-col justify-between rounded-[22px] p-6">
          <div>
            <div className="mb-4 flex items-center justify-between">
              <h3 className="text-xl font-bold text-slate-900 md:text-2xl">{obj.title}</h3>
              <div className="rounded-[12px] bg-cyan-50 p-3 text-cyan-700">{obj.icon}</div>
            </div>
            <p className="text-base leading-relaxed text-slate-600">{obj.detail}</p>
          </div>
        </div>
      ))}
    </div>
  )
}

function BenchmarkSlide({ slide }) {
  return (
    <div className="grid h-full gap-5 md:grid-cols-[1.25fr_0.75fr] overflow-hidden">
      {/* Main Table Card */}
      <div className="deck-card flex flex-col justify-between rounded-[22px] p-5 overflow-hidden">
        <div>
          <div className="mb-3 flex items-center justify-between border-b border-slate-100 pb-2.5">
            <div>
              <h3 className="text-lg font-bold text-slate-900 md:text-xl">OOD Evaluation Matrix</h3>
              <p className="text-[0.7rem] text-slate-500 mt-0.5">
                Key performance metrics generated across different detector/framework combinations.
              </p>
            </div>
            <div className="flex flex-col gap-1 items-end text-[0.68rem] font-semibold text-slate-500">
              <span className="rounded bg-slate-100 px-1.5 py-0.5 border border-slate-200/50">Dataset: OpenImages OOD (96 pics, max 3/class)</span>
              <span className="rounded bg-cyan-50 text-cyan-700 px-1.5 py-0.5 border border-cyan-150">Judge: Kimi VLM</span>
            </div>
          </div>
          
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="border-b border-slate-200 text-slate-550 font-bold uppercase tracking-[0.05em] text-[0.7rem]">
                  <th className="py-2.5 pr-4">Variant</th>
                  <th className="py-2.5 px-4 text-right">OOD Detection Rate</th>
                  <th className="py-2.5 pl-4 text-right">LLM Judge Accuracy</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 text-slate-700 text-sm">
                {slide.variants.map((v) => (
                  <tr key={v.name} className="hover:bg-slate-50/50">
                    <td className="py-3 pr-4 font-mono font-semibold text-slate-900">{v.name}</td>
                    <td className="py-3 px-4 text-right text-slate-900 font-semibold">{v.oodDetectionRate}</td>
                    <td className="py-3 pl-4 text-right text-slate-900 font-semibold">{v.llmJudgeAccuracy}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="mt-3 flex items-center justify-between rounded-[12px] bg-emerald-50/60 border border-emerald-250/50 p-3">
          <div className="flex items-center gap-2.5">
            <span className="flex h-2 w-2 rounded-full bg-emerald-500" />
            <p className="text-xs text-slate-700 font-semibold">
              Benchmark execution completed. Matrix results successfully populated.
            </p>
          </div>
          <span className="text-[0.68rem] text-slate-400 font-mono">Outputs: benchmark_matrix_summary.json</span>
        </div>
      </div>

      {/* Definition & Comparison Card */}
      <div className="deck-card flex flex-col justify-between rounded-[22px] p-5">
        <div className="space-y-3">
          <div>
            <h4 className="text-[0.7rem] font-bold uppercase tracking-[0.12em] text-cyan-805 mb-1.5">Evaluation Modes</h4>
            <div className="space-y-2">
              <div>
                <span className="inline-flex rounded-[6px] bg-cyan-50 text-cyan-700 border border-cyan-150 px-1.5 py-0.5 text-[0.68rem] font-bold font-mono">framework</span>
                <p className="mt-0.5 text-[0.68rem] text-slate-500 leading-normal">
                  Evaluates the full 5-layer pipeline: filters YOLO detections through outlier gates and enqueues candidates for asynchronous VLM verification.
                </p>
              </div>
              <div className="border-t border-slate-100 pt-2">
                <span className="inline-flex rounded-[6px] bg-slate-100 text-slate-600 border border-slate-200 px-1.5 py-0.5 text-[0.68rem] font-bold font-mono">only (detector-only)</span>
                <p className="mt-0.5 text-[0.68rem] text-slate-500 leading-normal">
                  Evaluates the detector in isolation. Outlier decisions are based solely on raw predictions of non-COCO classes, bypassing VLM verification.
                </p>
              </div>
            </div>
          </div>

          <div className="border-t border-slate-150 pt-2.5">
            <h4 className="text-[0.7rem] font-bold uppercase tracking-[0.12em] text-cyan-805 mb-1.5">Detector Models</h4>
            <div className="space-y-2">
              <div>
                <span className="text-[0.7rem] font-bold text-slate-900">YOLOv8n (Closed-Set)</span>
                <p className="mt-0.5 text-[0.68rem] text-slate-500 leading-normal">
                  Lightweight model limited to standard COCO classes (80 types). Relies heavily on confidence gating and VLM to detect outliers.
                </p>
              </div>
              <div className="border-t border-slate-100 pt-2">
                <span className="text-[0.7rem] font-bold text-slate-900">YOLO-World (Open-Vocabulary)</span>
                <p className="mt-0.5 text-[0.68rem] text-slate-500 leading-normal">
                  Capable of recognizing arbitrary classes at detection time via custom text prompts.
                </p>
              </div>
            </div>
          </div>
        </div>

        <div className="text-[0.68rem] text-slate-400 pt-2 border-t border-slate-100 font-mono">
          Focus: Known vs. Out-of-Distribution (OOD)
        </div>
      </div>
    </div>
  )
}

export default App
