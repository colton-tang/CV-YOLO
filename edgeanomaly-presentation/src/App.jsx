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
import architectureGraphic from '../../assets/P1.png'
import smartReviewGraphic from '../../assets/P4.png'

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
    id: 'gates',
    type: 'gates',
    header: '02. OOD Decision Flow',
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
        body: 'If confidence exceeds HIGH_CONFIDENCE_THRESHOLD (0.7) and the class is known, the object is treated as in-distribution and auto-passes to RESOLVED.',
      },
      {
        title: 'Gate 3: OOD candidate',
        body: 'Tracks in the uncertainty band (0.3 - 0.7) or outside known_classes are treated as potential OOD objects, enter VERIFYING, and are queued for vision-language review once.',
      },
      {
        title: 'Below low threshold',
        body: 'Detections below LOW_CONFIDENCE_THRESHOLD (0.3) are ignored so the OOD path is reserved for meaningful candidates rather than noise.',
      },
    ],
  },
  {
    id: 'async',
    type: 'async',
    header: '03. Smart Review for Unusual Objects',
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
    header: '04. Performance Benchmarks',
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
        oodDetectionRate: '92.71%',
        llmJudgeAccuracy: '92.71%',
      },
    ],
  },
  {
    id: 'future',
    type: 'future',
    header: '05. Future Development',
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

  const exportSlidesToPdf = async () => {
    if (isExporting || !exportSlideRefs.current.length) return
    setIsExporting(true)

    try {
      const [{ toJpeg }, { jsPDF }] = await Promise.all([import('html-to-image'), import('jspdf')])
      const exportWidth = 1600
      const exportHeight = 900
      const pageImages = []

      if (document.fonts?.ready) {
        await document.fonts.ready
      }

      for (let slideIndex = 0; slideIndex < exportSlideRefs.current.length; slideIndex += 1) {
        const slideElement = exportSlideRefs.current[slideIndex]
        if (!slideElement) continue

        const dataUrl = await toJpeg(slideElement, {
          cacheBust: true,
          pixelRatio: 1.5,
          quality: 0.82,
          backgroundColor: '#f8fafc',
          width: exportWidth,
          height: exportHeight,
          canvasWidth: Math.round(exportWidth * 1.5),
          canvasHeight: Math.round(exportHeight * 1.5),
        })

        pageImages.push({
          dataUrl,
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

        pdf.addImage(page.dataUrl, 'JPEG', 0, 0, 1600, 900, undefined, 'MEDIUM')
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
    <div className="deck-shell text-slate-900" data-total-slides={slides.length}>
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
    : 'slide-panel diagram-grid mx-auto aspect-[16/9] min-h-[580px] w-full max-w-7xl rounded-[28px] p-6 shadow-panel backdrop-blur md:p-8 flex flex-col overflow-hidden'

  return (
    <section ref={slideRef} className={shellClass}>
      {slide.type === 'cover' ? (
        <CoverSlide slide={slide} isExportMode={isExportMode} />
      ) : (
        <StandardSlide slide={slide} slideIndex={slideIndex} isExportMode={isExportMode} />
      )}
    </section>
  )
}

function CoverSlide({ slide, isExportMode = false }) {
  return (
    <div className="grid h-full gap-8 lg:grid-cols-[1.15fr_0.85fr] items-center overflow-hidden py-2">
      {/* Left Column: Title & Presenter Info */}
      <div className="flex flex-col justify-between h-full space-y-6">
        <div>
          {/* Glowing Kicker */}
          <div className="inline-flex items-center gap-2 rounded-full border border-cyan-200/50 bg-cyan-50/50 px-3 py-1 text-xs font-bold uppercase tracking-[0.2em] text-cyan-800">
            <span className={`flex h-2 w-2 rounded-full bg-cyan-500 ${isExportMode ? '' : 'animate-pulse'}`} />
            {slide.kicker}
          </div>
          
          {/* Display Title with Gradient */}
          <h1 className="display-serif mt-5 text-[3.8rem] leading-[0.95] tracking-tight text-slate-900 md:text-[4.5rem]">
            Edge Anomaly <br />
            <span className={isExportMode ? 'text-cyan-700' : 'bg-gradient-to-r from-cyan-600 via-teal-600 to-amber-500 bg-clip-text text-transparent'}>
              Detection Framework
            </span>
          </h1>
          
          <p className="mt-4 max-w-2xl text-[1.08rem] leading-relaxed text-slate-600 md:text-[1.2rem]">
            An intelligent edge surveillance pipeline that fast-paths known in-distribution objects and escalates Out-of-Distribution candidates for deeper vision-language verification.
          </p>
        </div>

        {/* Presenter Profile Box */}
        <div className="border-t border-slate-200/80 pt-5">
          <p className="text-[0.7rem] font-bold uppercase tracking-[0.2em] text-slate-400">Presented By</p>
          <div className="mt-3 flex items-start gap-4">
            <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-cyan-700 to-teal-800 text-lg font-bold text-white shadow-md">
              QT
            </div>
            <div>
              <h3 className="text-xl font-bold text-slate-900 leading-none">Qinxing Tang</h3>
              <p className="mt-1 text-sm font-semibold text-cyan-800 leading-tight">
                M.Sc. Candidate, Financial Engineering
              </p>
              <p className="text-xs text-slate-500 mt-0.5">
                National University of Singapore • <span className="font-mono">Qinxing_tang@u.nus.edu</span>
              </p>
              <p className="text-xs text-slate-500 mt-0.5">
                GitHub: <a href="https://github.com/colton-tang/CV-YOLO.git" target="_blank" rel="noopener noreferrer" className="font-mono text-cyan-700 hover:underline">https://github.com/colton-tang/CV-YOLO.git</a>
              </p>
              <p className="text-[0.7rem] text-slate-400 mt-1 uppercase font-bold tracking-wider">
                19th June 2026
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Right Column: Visual Pipeline Flow Graphic */}
      <div
        className={`deck-card relative rounded-[28px] p-6 text-slate-800 h-full flex flex-col justify-between overflow-hidden ${
          isExportMode ? '' : 'shadow-panel backdrop-blur'
        }`}
      >
        {/* Subtle glowing radial background graphics */}
        {!isExportMode && <div className="absolute -right-20 -top-20 h-64 w-64 rounded-full bg-cyan-500/5 blur-[80px] pointer-events-none" />}
        {!isExportMode && <div className="absolute -left-20 -bottom-20 h-64 w-64 rounded-full bg-amber-500/5 blur-[80px] pointer-events-none" />}

        <div className="relative">
          <p className="section-label text-cyan-700">5-Layer Pipeline Flow</p>
          <p className="mt-1 text-xs text-slate-500">Unified frame stream normalizer to status-aware alert rendering.</p>
          
          <div className="mt-5 space-y-2.5">
            {[
              { num: 'L1', name: 'Ingestion', desc: 'Unified camera/file stream normalizer' },
              { num: 'L2', name: 'Detection & Tracking', desc: 'YOLO + BotSort track history' },
              { num: 'L3', name: 'OOD Gate Filter', desc: 'Confidence boundary check (0.3 - 0.7)', glow: true },
              { num: 'L4', name: 'Async OOD Verifier', desc: 'Background Qwen-VL review' },
              { num: 'L5', name: 'Render & Alert', desc: 'Operator UI annotation & alerts' }
            ].map((step, index) => (
              <div 
                key={step.name} 
                className={`relative flex items-center gap-3 rounded-[14px] border p-2 transition ${
                  step.glow 
                    ? 'border-cyan-400 bg-cyan-50/70 shadow-[0_0_12px_rgba(6,182,212,0.1)]' 
                    : 'border-slate-100 bg-white/70 shadow-sm'
                }`}
              >
                {/* Connecting Line */}
                {index < 4 && (
                  <div className="absolute left-[21px] top-[30px] w-[2px] h-[16px] bg-slate-200" />
                )}

                <div className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-[8px] text-[0.7rem] font-bold ${
                  step.glow 
                    ? 'bg-cyan-600 text-white shadow-[0_0_10px_rgba(8,145,178,0.3)]' 
                    : 'bg-slate-100 text-slate-600 border border-slate-200'
                }`}>
                  {step.num}
                </div>
                
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between">
                    <h4 className="text-[0.72rem] font-bold text-slate-800 leading-none">{step.name}</h4>
                    {step.glow && (
                      <span className="rounded-full bg-cyan-100/80 px-1.5 py-0.5 text-[0.55rem] font-bold text-cyan-800 border border-cyan-200/50 uppercase tracking-wider">
                        OOD Decision Boundary
                      </span>
                    )}
                  </div>
                  <p className="mt-0.5 text-[0.66rem] leading-none text-slate-500 break-words truncate">{step.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Footnote */}
        <div className="relative border-t border-slate-100 pt-3 mt-4 flex items-center justify-between text-[0.68rem] text-slate-500">
          <span>Project: EdgeAnomalyCCTV</span>
          <span className="font-mono text-cyan-700">main.py runtime</span>
        </div>
      </div>
    </div>
  )
}

function StandardSlide({ slide, slideIndex, isExportMode }) {
  return (
    <div className="flex h-full flex-col">
      <div className="mb-4 md:mb-5 flex items-start justify-between gap-4">
        <div className="max-w-4xl">
          <p className="section-label mb-1">{slide.tagline}</p>
          <h2 className={`${isExportMode ? 'text-[3.6rem]' : 'text-[1.8rem] md:text-[2.5rem]'} display-serif leading-[0.93] text-slate-900`}>
            {slide.header}
          </h2>
        </div>
        <div className="rounded-[18px] border border-slate-200 bg-[rgba(255,252,246,0.85)] px-4 py-2 text-2xl md:text-3xl font-black text-slate-200">
          0{slideIndex}
        </div>
      </div>

      <div className="flex-1 min-h-0">
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
    <div className="deck-card relative h-full overflow-hidden rounded-[24px] border border-slate-200 bg-white shadow-[0_20px_44px_rgba(18,35,58,0.08)]">
      <img
        src={architectureGraphic}
        alt="Five-layer architecture system map"
        className="h-full w-full object-contain object-center"
      />
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
    <div className="grid h-full gap-4 md:grid-cols-[1.05fr_0.95fr] overflow-hidden">
      {/* Left Column: Detector Output */}
      <div className="deck-card flex flex-col rounded-[22px] p-4 overflow-hidden h-full">
        <div className="mb-2 flex items-start justify-between gap-4 shrink-0">
          <div>
            <p className="section-label">Annotated Output</p>
            <h3 className="mt-1 text-[1.1rem] font-bold text-slate-900 leading-snug">Base yolov8n result on bus.jpg</h3>
            <p className="text-[0.75rem] text-slate-500 mt-0.5">{slide.example.subtitle}</p>
          </div>
          <span className="rounded-[8px] border border-emerald-200 bg-emerald-50 px-2 py-0.5 text-[0.7rem] font-bold text-emerald-700 shrink-0">
            Resolved
          </span>
        </div>

        <div className="flex-1 min-h-0 overflow-hidden rounded-[14px] border border-slate-200 bg-slate-50 flex items-center justify-center p-1.5">
          <img
            src={slide.example.image}
            alt="Annotated detector output showing a bus and three people labeled as resolved."
            className="h-full max-w-full object-contain object-center"
          />
        </div>

        <div className="mt-2.5 grid gap-2 md:grid-cols-3 shrink-0">
          {slide.example.facts.map((fact) => (
            <div key={fact} className="rounded-[10px] border border-slate-150 bg-slate-50/70 px-2 py-1 text-[0.7rem] font-semibold text-slate-600 text-center">
              {fact}
            </div>
          ))}
        </div>
      </div>

      {/* Right Column: Outcomes & Gates */}
      <div className="flex h-full flex-col gap-3 overflow-hidden">
        {/* State Outcomes & Thresholds */}
        <div className="deck-card rounded-[22px] p-3 grid grid-cols-[1fr_1.4fr] gap-3">
          <div>
            <div className="mb-1.5 flex items-center gap-2">
              <ShieldCheck className="text-cyan-700" size={16} />
              <h3 className="text-xs font-bold uppercase tracking-[0.12em] text-cyan-805">State outcomes</h3>
            </div>
            <div className="grid gap-1 grid-cols-2">
              {slide.statuses.map((status) => (
                <div key={status} className={`rounded-[8px] border px-2 py-0.5 text-center text-[0.7rem] font-bold ${STATUS_STYLES[status]}`}>
                  {status}
                </div>
              ))}
            </div>
          </div>

          <div className="border-l border-slate-200/80 pl-3">
            <div className="mb-1.5 flex items-center gap-2">
              <SquareDashedMousePointer className="text-amber-600" size={14} />
              <h3 className="text-xs font-bold uppercase tracking-[0.12em] text-cyan-805">Thresholds</h3>
            </div>
            <div className="space-y-1 text-[0.56rem] md:text-[0.62rem]">
              <div className="flex justify-between items-center rounded-[8px] border border-slate-200/60 bg-slate-50/50 px-2 py-0.5">
                <span className="font-mono text-slate-500">LOW_CONFIDENCE_THRESHOLD</span>
                <span className="font-mono font-bold text-amber-700 ml-2">0.3</span>
              </div>
              <div className="flex justify-between items-center rounded-[8px] border border-slate-200/60 bg-slate-50/50 px-2 py-0.5">
                <span className="font-mono text-slate-500">HIGH_CONFIDENCE_THRESHOLD</span>
                <span className="font-mono font-bold text-emerald-700 ml-2">0.7</span>
              </div>
            </div>
          </div>
        </div>

        {/* Gates Grid */}
        <div className="grid flex-1 gap-2 md:grid-cols-2 min-h-0">
          {slide.gates.map((gate) => (
            <div key={gate.title} className="deck-card rounded-[18px] p-3.5 flex flex-col justify-between overflow-hidden">
              <div>
                <div className="mb-1.5 flex items-center gap-2">
                  <SquareDashedMousePointer className="shrink-0 text-amber-600" size={15} />
                  <h3 className="text-[0.84rem] font-bold text-slate-900 leading-snug">{gate.title}</h3>
                </div>
                <p className="text-[0.76rem] leading-relaxed text-slate-600">{gate.body}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

function AsyncSlide({ slide }) {
  return (
    <div className="grid h-full gap-4 md:grid-cols-[1.24fr_0.76fr] overflow-hidden">
      <div className="deck-card flex flex-col rounded-[22px] p-4 overflow-hidden">
        <div>
          <div className="mb-3 flex items-center justify-between gap-3 border-b border-slate-100 pb-2">
            <div className="flex items-center gap-2">
              <ListTree className="text-cyan-700" size={18} />
              <h3 className="text-base font-bold text-slate-900 md:text-lg">Smart review sequence</h3>
            </div>
            <span className="rounded-[999px] border border-cyan-200 bg-cyan-50 px-2.5 py-1 text-[0.62rem] font-bold uppercase tracking-[0.16em] text-cyan-800">
              Selective escalation
            </span>
          </div>
          <div className="overflow-hidden rounded-[18px] border border-slate-200 bg-white shadow-[0_16px_36px_rgba(18,35,58,0.08)]">
            <img
              src={smartReviewGraphic}
              alt="Four-step smart review sequence for unusual objects"
              className="h-full w-full object-cover object-center"
            />
          </div>
        </div>

        <div className="mt-3 grid gap-2 md:grid-cols-4">
          {slide.flow.map((step, index) => (
            <div key={step.name} className="rounded-[14px] border border-slate-200/80 bg-slate-50/80 p-2.5">
              <div className="mb-1.5 flex items-center gap-2">
                <div className="flex h-6 w-6 shrink-0 items-center justify-center rounded-[7px] bg-cyan-700 text-[0.7rem] font-bold text-white">
                  {index + 1}
                </div>
                <h4 className="text-[0.72rem] font-bold leading-tight text-slate-900">{step.name}</h4>
              </div>
              <p className="text-[0.66rem] leading-snug text-slate-500">{step.detail}</p>
            </div>
          ))}
        </div>
      </div>

      <div className="deck-card flex flex-col justify-between rounded-[22px] p-4 overflow-hidden">
        <div className="space-y-3.5">
          <div>
            <div className="mb-1.5 flex items-center gap-2">
              <Cpu className="text-cyan-700" size={16} />
              <h3 className="text-xs font-bold uppercase tracking-[0.12em] text-cyan-800">Why this matters</h3>
            </div>
            <p className="text-[0.78rem] leading-relaxed text-slate-600">{slide.summary}</p>
          </div>

          <div className="rounded-[16px] border border-emerald-200 bg-emerald-50/70 p-3">
            <p className="text-[0.68rem] font-bold uppercase tracking-[0.12em] text-emerald-700">Operational takeaway</p>
            <p className="mt-1.5 text-[0.74rem] leading-relaxed text-emerald-900">
              The operator keeps a live view while the system quietly double-checks only the detections that look unfamiliar.
            </p>
          </div>

          <div className="border-t border-slate-200 pt-2.5">
            <p className="mb-2 text-[0.68rem] font-bold uppercase tracking-[0.12em] text-slate-400">Stakeholder value</p>
            <div className="grid gap-1.5 grid-cols-1">
              {slide.benefits.map((benefit) => (
                <div
                  key={benefit}
                  className="rounded-[10px] border border-slate-200 bg-slate-50/70 px-2.5 py-2 text-[0.72rem] font-semibold text-slate-600"
                >
                  {benefit}
                </div>
              ))}
            </div>
          </div>

          <div className="border-t border-slate-200 pt-2.5">
            <p className="mb-2 text-[0.68rem] font-bold uppercase tracking-[0.12em] text-slate-400">Review trigger</p>
            <div className="rounded-[14px] border border-amber-200 bg-amber-50/80 p-3">
              <p className="text-[0.74rem] leading-relaxed text-slate-700">
                Objects that are low-confidence, ambiguous, or outside the known class set are escalated into background verification.
              </p>
            </div>
          </div>
        </div>

        <div className="deck-card-dark mt-3 rounded-[10px] p-2 text-[0.7rem] leading-relaxed text-slate-200 text-center">
          Unusual objects get extra scrutiny, but the rest of the scene keeps moving.
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
              <span className="rounded bg-slate-100 px-1.5 py-0.5 border border-slate-200/50">Dataset: OpenImages OOD (96 pics)</span>
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
