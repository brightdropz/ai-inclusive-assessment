
import { useEffect, useState, type FormEvent } from 'react'
import './App.css'

// ============================================================
// CONFIGURATION
// ============================================================

// Local development URL. We can configure the public API URL
// later using VITE_API_BASE_URL during deployment.
const API_BASE =
  import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'

// ============================================================
// TYPES
// ============================================================

type View =
  | 'overview'
  | 'assessment'
  | 'accessibility'
  | 'results'
  | 'ai-prediction'

type Difficulty = 'Easy' | 'Medium' | 'Hard'

type Question = {
  id: number
  category: string
  difficulty: Difficulty
  prompt: string
  options: string[]
}

type StartResponse = {
  session_id: string
  question_number: number
  total_questions: number
  ability_level: Difficulty
  question: Question
}

type AnswerResponse = {
  correct: boolean
  explanation: string
  score: number
  answered_count: number
  accuracy: number
  ability_level: Difficulty
  finished: boolean
  next_question: Question | null
}

type AssessmentResults = {
  session_id: string
  user_id: string
  score: number
  answered_count: number
  accuracy: number
  ability_level: Difficulty
  finished: boolean
  created_at: string
}

type Preferences = {
  large_text: boolean
  high_contrast: boolean
  reduced_motion: boolean
}

type StudentInputs = {
  studytime: number
  failures: number
  schoolsup: string
  famsup: string
  paid: string
  activities: string
  nursery: string
  higher: string
  internet: string
  traveltime: number
  freetime: number
  goout: number
  Dalc: number
  Walc: number
  health: number
  absences: number
}

type NumericField =
  | 'studytime'
  | 'failures'
  | 'traveltime'
  | 'freetime'
  | 'goout'
  | 'Dalc'
  | 'Walc'
  | 'health'
  | 'absences'

type YesNoField =
  | 'schoolsup'
  | 'famsup'
  | 'paid'
  | 'activities'
  | 'nursery'
  | 'higher'
  | 'internet'

type PredictionResult = {
  model: string
  prediction: string
  predicted_class: string
  fail_probability: number
  pass_probability: number
  risk_threshold: number
  needs_additional_review: boolean
  notice: string
}

type DatasetInfo = {
  dataset_name: string
  source: string
  records: number
  raw_columns: number
  missing_values: number
  selected_ml_features: number
  target: string
  grade_columns: string[]
  file: string
  license: string
}

type ModelInfo = {
  model: string
  dataset: string
  input_features: number
  feature_columns: string[]
  target: string
  fail_probability_threshold: number
  status: string
  purpose: string
}

type Feedback = {
  correct: boolean
  explanation: string
  finished: boolean
  nextQuestion: Question | null
}

// ============================================================
// API HELPER
// ============================================================

async function apiRequest<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const headers = new Headers(options.headers)
  headers.set('Content-Type', 'application/json')

  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers,
  })

  const data = await response.json().catch(() => ({}))

  if (!response.ok) {
    throw new Error(
      data.detail || `Request failed: ${response.status}`,
    )
  }

  return data as T
}

// ============================================================
// DEFAULT ACCESSIBILITY PREFERENCES
// ============================================================

const defaultPreferences: Preferences = {
  large_text: false,
  high_contrast: false,
  reduced_motion: false,
}

// ============================================================
// STUDENT RISK PREDICTION PAGE
// This uses the actual trained Random Forest API.
// ============================================================

const initialStudentInputs: StudentInputs = {
  studytime: 3,
  failures: 0,
  schoolsup: 'no',
  famsup: 'yes',
  paid: 'no',
  activities: 'yes',
  nursery: 'yes',
  higher: 'yes',
  internet: 'yes',
  traveltime: 1,
  freetime: 3,
  goout: 2,
  Dalc: 1,
  Walc: 2,
  health: 3,
  absences: 2,
}

function StudentRiskPrediction() {
  const [form, setForm] = useState<StudentInputs>(
    initialStudentInputs,
  )

  const [result, setResult] = useState<PredictionResult | null>(null)
  const [datasetInfo, setDatasetInfo] = useState<DatasetInfo | null>(null)
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null)

  const [loading, setLoading] = useState(false)
  const [loadingInfo, setLoadingInfo] = useState(true)
  const [error, setError] = useState('')

  // Retrieve real dataset and model information from FastAPI.
  useEffect(() => {
    let active = true

    async function loadInformation() {
      try {
        const [dataset, model] = await Promise.all([
          apiRequest<DatasetInfo>('/api/dataset/info'),
          apiRequest<ModelInfo>('/api/model/info'),
        ])

        if (active) {
          setDatasetInfo(dataset)
          setModelInfo(model)
        }
      } catch (err) {
        if (active) {
          setError(
            err instanceof Error
              ? err.message
              : 'Could not load model information.',
          )
        }
      } finally {
        if (active) {
          setLoadingInfo(false)
        }
      }
    }

    loadInformation()

    return () => {
      active = false
    }
  }, [])

  const updateNumber = (
    field: NumericField,
    value: string,
  ) => {
    setForm(previous => ({
      ...previous,
      [field]: Number(value),
    }))
  }

  const updateChoice = (
    field: YesNoField,
    value: string,
  ) => {
    setForm(previous => ({
      ...previous,
      [field]: value,
    }))
  }

  const numberField = (
    field: NumericField,
    label: string,
    min: number,
    max: number,
  ) => (
    <label className="prediction-field" key={field}>
      <span>{label}</span>
      <select
        value={form[field]}
        onChange={event =>
          updateNumber(field, event.target.value)
        }
      >
        {Array.from(
          { length: max - min + 1 },
          (_, index) => min + index,
        ).map(value => (
          <option key={value} value={value}>
            {value}
          </option>
        ))}
      </select>
    </label>
  )

  const yesNoField = (
    field: YesNoField,
    label: string,
  ) => (
    <label className="prediction-field" key={field}>
      <span>{label}</span>
      <select
        value={form[field]}
        onChange={event =>
          updateChoice(field, event.target.value)
        }
      >
        <option value="yes">Yes</option>
        <option value="no">No</option>
      </select>
    </label>
  )

  const submitPrediction = async (
    event: FormEvent<HTMLFormElement>,
  ) => {
    event.preventDefault()

    setLoading(true)
    setError('')
    setResult(null)

    try {
      const data = await apiRequest<PredictionResult>(
        '/api/predictions/student-risk',
        {
          method: 'POST',
          body: JSON.stringify(form),
        },
      )

      setResult(data)
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'Unable to generate prediction.',
      )
    } finally {
      setLoading(false)
    }
  }

  const resetPrediction = () => {
    setForm(initialStudentInputs)
    setResult(null)
    setError('')
  }

  return (
    <section className="page-content prediction-content">
      <div className="eyebrow">MACHINE LEARNING MODULE</div>

      <h1>Student performance insights.</h1>

      <p className="page-subtitle">
        Explore an experimental Random Forest prediction using
        selected educational and learning-related factors.
      </p>

      <div className="prediction-intro">
        <div className="prediction-symbol">✳</div>

        <div>
          <strong>Random Forest Classifier</strong>
          <p>
            An experimental educational prediction model
            trained using the UCI Student Performance dataset.
          </p>
        </div>
      </div>

      {/* DATASET INFORMATION */}

      <div className="prediction-form-card">
        <div className="prediction-section-heading">
          <div>
            <h2>Dataset information</h2>
            <p>Verified information from the backend</p>
          </div>
          <span>DATA</span>
        </div>

        {loadingInfo ? (
          <p>Loading dataset information...</p>
        ) : datasetInfo ? (
          <>
            <div className="prediction-data-grid">
              <div>
                <strong>{datasetInfo.records}</strong>
                <span>Student records</span>
              </div>

              <div>
                <strong>{datasetInfo.raw_columns}</strong>
                <span>Raw CSV columns</span>
              </div>

              <div>
                <strong>{datasetInfo.selected_ml_features}</strong>
                <span>ML input features</span>
              </div>

              <div>
                <strong>{datasetInfo.missing_values}</strong>
                <span>Missing values</span>
              </div>
            </div>

            <p className="prediction-data-description">
              {datasetInfo.dataset_name}
            </p>

            <p className="prediction-data-description">
              Target: {datasetInfo.target}
            </p>

            <a
              href={datasetInfo.source}
              target="_blank"
              rel="noreferrer"
              className="prediction-source-link"
            >
              View official UCI dataset ↗
            </a>
          </>
        ) : (
          <p>Dataset information is currently unavailable.</p>
        )}
      </div>

      {/* MODEL INFORMATION */}

      <div className="prediction-form-card">
        <div className="prediction-section-heading">
          <div>
            <h2>Model information</h2>
            <p>Trained model details</p>
          </div>
          <span>AI</span>
        </div>

        {modelInfo ? (
          <div className="prediction-model-info">
            <div>
              <span>Algorithm</span>
              <strong>{modelInfo.model}</strong>
            </div>

            <div>
              <span>Input features</span>
              <strong>{modelInfo.input_features}</strong>
            </div>

            <div>
              <span>Risk probability threshold</span>
              <strong>
                {(modelInfo.fail_probability_threshold * 100).toFixed(1)}%
              </strong>
            </div>

            <div>
              <span>Model status</span>
              <strong className="prediction-model-status">
                {modelInfo.status}
              </strong>
            </div>
          </div>
        ) : (
          <p>
            Model information will appear when the backend is available.
          </p>
        )}
      </div>

      {/* PREDICTION INPUT FORM */}

      <form onSubmit={submitPrediction}>
        <div className="prediction-form-card">
          <div className="prediction-section-heading">
            <div>
              <h2>Learning profile</h2>
              <p>Academic and study-related information</p>
            </div>
            <span>01</span>
          </div>

          <div className="prediction-fields">
            {numberField('studytime', 'Weekly study time', 1, 4)}
            {numberField('failures', 'Previous class failures', 0, 3)}
            {numberField('traveltime', 'Travel time to school', 1, 4)}
            {numberField('absences', 'Number of absences', 0, 100)}
          </div>
        </div>

        <div className="prediction-form-card">
          <div className="prediction-section-heading">
            <div>
              <h2>Learning support</h2>
              <p>Educational assistance and available resources</p>
            </div>
            <span>02</span>
          </div>

          <div className="prediction-fields">
            {yesNoField('schoolsup', 'Extra school support')}
            {yesNoField('famsup', 'Family educational support')}
            {yesNoField('paid', 'Additional paid classes')}
            {yesNoField('higher', 'Plans for higher education')}
            {yesNoField('internet', 'Internet access')}
            {yesNoField('nursery', 'Attended nursery school')}
            {yesNoField('activities', 'Extracurricular activities')}
          </div>
        </div>

        <div className="prediction-form-card">
          <div className="prediction-section-heading">
            <div>
              <h2>Learning and lifestyle factors</h2>
              <p>Self-reported factors included in the dataset</p>
            </div>
            <span>03</span>
          </div>

          <div className="prediction-fields">
            {numberField('freetime', 'Free time (1–5)', 1, 5)}
            {numberField('goout', 'Going out with friends (1–5)', 1, 5)}
            {numberField('Dalc', 'Weekday consumption (1–5)', 1, 5)}
            {numberField('Walc', 'Weekend consumption (1–5)', 1, 5)}
            {numberField('health', 'Self-reported health (1–5)', 1, 5)}
          </div>
        </div>

        <div className="prediction-actions">
          <button
            type="button"
            className="prediction-reset"
            onClick={resetPrediction}
            disabled={loading}
          >
            Reset fields
          </button>

          <button
            type="submit"
            className="primary-button"
            disabled={loading || !modelInfo}
          >
            {loading ? 'Analysing...' : 'Generate prediction'}
            <span>→</span>
          </button>
        </div>
      </form>

      {error && (
        <div className="prediction-error" role="alert">
          <strong>Notice</strong>
          <p>{error}</p>
        </div>
      )}

      {/* RANDOM FOREST PREDICTION RESULTS */}

      {result && (
        <div className="prediction-result" aria-live="polite">
          <div className="eyebrow">MODEL OUTPUT</div>

          <h2>Prediction summary</h2>

          <div className="prediction-outcome">
            <span className="outcome-label">
              Predicted category
            </span>

            <strong
              className={
                result.needs_additional_review
                  ? 'outcome-risk'
                  : 'outcome-pass'
              }
            >
              {result.predicted_class}
            </strong>

            <p>{result.prediction}</p>
          </div>

          <div className="probability-grid">
            <div className="probability-item">
              <span>Pass probability</span>

              <strong>
                {result.pass_probability.toFixed(2)}%
              </strong>

              <div className="probability-track">
                <div
                  className="probability-pass"
                  style={{
                    width: `${result.pass_probability}%`,
                  }}
                />
              </div>
            </div>

            <div className="probability-item">
              <span>Fail probability</span>

              <strong>
                {result.fail_probability.toFixed(2)}%
              </strong>

              <div className="probability-track">
                <div
                  className="probability-fail"
                  style={{
                    width: `${result.fail_probability}%`,
                  }}
                />
              </div>
            </div>
          </div>

          <div className="prediction-explanation">
            <strong>Prediction details</strong>

            <p>Algorithm: {result.model}</p>

            <p>
              Risk threshold:{' '}
              {(result.risk_threshold * 100).toFixed(1)}%
            </p>

            <p>
              Additional review:{' '}
              {result.needs_additional_review
                ? 'Recommended'
                : 'Not triggered'}
            </p>
          </div>

          <div className="prediction-disclaimer">
            <strong>Important limitation</strong>
            <p>{result.notice}</p>
          </div>
        </div>
      )}

      <p className="disclaimer">
        This experimental model uses historical educational data.
        Its predictions must not be treated as a diagnosis or
        substitute for teacher judgement.
      </p>
    </section>
  )
}

// ============================================================
// MAIN APPLICATION
// ============================================================

function App() {
  const [view, setView] = useState<View>('overview')

  // Assessment state
  const [sessionId, setSessionId] = useState('')
  const [question, setQuestion] = useState<Question | null>(null)
  const [selected, setSelected] = useState<number | null>(null)
  const [feedback, setFeedback] = useState<Feedback | null>(null)

  const [score, setScore] = useState(0)
  const [answeredCount, setAnsweredCount] = useState(0)
  const [ability, setAbility] = useState<Difficulty>('Medium')

  const [results, setResults] =
    useState<AssessmentResults | null>(null)

  // Accessibility settings
  const [preferences, setPreferences] =
    useState<Preferences>(defaultPreferences)

  // General interface state
  const [busy, setBusy] = useState(false)
  const [apiError, setApiError] = useState('')
  const [settingsMessage, setSettingsMessage] = useState('')

  // ==========================================================
  // START ASSESSMENT
  // ==========================================================

  const startAssessment = async () => {
    setBusy(true)
    setApiError('')

    try {
      const data = await apiRequest<StartResponse>(
        '/api/sessions/start',
        {
          method: 'POST',
          body: JSON.stringify({
            user_id: 'demo-user',
          }),
        },
      )

      setSessionId(data.session_id)
      setQuestion(data.question)
      setSelected(null)
      setFeedback(null)
      setScore(0)
      setAnsweredCount(0)
      setAbility(data.ability_level)
      setResults(null)

      setView('assessment')
    } catch (error) {
      setApiError(
        error instanceof Error
          ? error.message
          : 'Could not start assessment.',
      )
    } finally {
      setBusy(false)
    }
  }

  // ==========================================================
  // SUBMIT ANSWER
  // ==========================================================

  const submitAnswer = async () => {
    if (
      selected === null ||
      !question ||
      !sessionId ||
      feedback
    ) {
      return
    }

    setBusy(true)
    setApiError('')

    try {
      const data = await apiRequest<AnswerResponse>(
        `/api/sessions/${sessionId}/answers`,
        {
          method: 'POST',
          body: JSON.stringify({
            question_id: question.id,
            selected_option: selected,
          }),
        },
      )

      setScore(data.score)
      setAnsweredCount(data.answered_count)
      setAbility(data.ability_level)

      setFeedback({
        correct: data.correct,
        explanation: data.explanation,
        finished: data.finished,
        nextQuestion: data.next_question,
      })

      // Retrieve the final result after the assessment ends.
      if (data.finished) {
        try {
          const finalResults =
            await apiRequest<AssessmentResults>(
              `/api/sessions/${sessionId}/results`,
            )

          setResults(finalResults)
        } catch {
          setApiError(
            'Your answer was recorded, but final results could not be loaded.',
          )
        }
      }
    } catch (error) {
      setApiError(
        error instanceof Error
          ? error.message
          : 'Could not submit answer.',
      )
    } finally {
      setBusy(false)
    }
  }

  // ==========================================================
  // NEXT QUESTION
  // ==========================================================

  const nextQuestion = () => {
    if (!feedback) return

    if (feedback.finished) {
      setView('results')
      return
    }

    if (feedback.nextQuestion) {
      setQuestion(feedback.nextQuestion)
      setSelected(null)
      setFeedback(null)
    }
  }

  // ==========================================================
  // ACCESSIBILITY PREFERENCES
  // ==========================================================

  const openAccessibility = async () => {
    setView('accessibility')
    setApiError('')
    setSettingsMessage('')

    try {
      const data = await apiRequest<Preferences>(
        '/api/users/demo-user/preferences',
      )

      setPreferences(data)
    } catch (error) {
      setApiError(
        error instanceof Error
          ? error.message
          : 'Could not load preferences.',
      )
    }
  }

  const savePreferences = async () => {
    setBusy(true)
    setApiError('')
    setSettingsMessage('')

    try {
      const data = await apiRequest<Preferences>(
        '/api/users/demo-user/preferences',
        {
          method: 'POST',
          body: JSON.stringify(preferences),
        },
      )

      setPreferences(data)

      setSettingsMessage(
        'Your accessibility preferences have been saved.',
      )
    } catch (error) {
      setApiError(
        error instanceof Error
          ? error.message
          : 'Could not save preferences.',
      )
    } finally {
      setBusy(false)
    }
  }

  // ==========================================================
  // READ QUESTION ALOUD
  // ==========================================================

  const readQuestion = () => {
    if (!question || !('speechSynthesis' in window)) {
      setApiError(
        'Speech playback is not supported by this browser.',
      )
      return
    }

    window.speechSynthesis.cancel()

    window.speechSynthesis.speak(
      new SpeechSynthesisUtterance(question.prompt),
    )
  }

  const scorePercent = results
    ? results.accuracy
    : answeredCount
      ? Math.round((score / answeredCount) * 100)
      : 0

  // ==========================================================
  // APPLICATION LAYOUT
  // ==========================================================

  return (
    <div
      className={`app-shell ${
        preferences.high_contrast ? 'high-contrast' : ''
      } ${
        preferences.large_text ? 'large-text' : ''
      } ${
        preferences.reduced_motion ? 'reduced-motion' : ''
      }`}
    >
      {/* SIDEBAR */}

      <aside className="sidebar">
        <div className="brand">
          <div className="brand-symbol">A</div>

          <div>
            <strong>INCLUSIFY</strong>
            <span>AI ASSESSMENT</span>
          </div>
        </div>

        <div className="nav-label">WORKSPACE</div>

        <nav className="side-nav" aria-label="Main navigation">
          <button
            className={
              view === 'overview'
                ? 'nav-item active'
                : 'nav-item'
            }
            onClick={() => setView('overview')}
          >
            <span>⌂</span>
            Overview
          </button>

          <button
            className={
              view === 'assessment'
                ? 'nav-item active'
                : 'nav-item'
            }
            onClick={startAssessment}
            disabled={busy}
          >
            <span>▤</span>
            Take assessment
          </button>

          <button
            className={
              view === 'results'
                ? 'nav-item active'
                : 'nav-item'
            }
            onClick={() => setView('results')}
          >
            <span>▥</span>
            My results
          </button>

          <button
            className={
              view === 'ai-prediction'
                ? 'nav-item active'
                : 'nav-item'
            }
            onClick={() => {
              setApiError('')
              setView('ai-prediction')
            }}
          >
            <span>✦</span>
            AI Prediction
          </button>

          <button
            className={
              view === 'accessibility'
                ? 'nav-item active'
                : 'nav-item'
            }
            onClick={openAccessibility}
          >
            <span>◉</span>
            Accessibility
          </button>
        </nav>

        <div className="sidebar-bottom">
          <div className="support-card">
            <span className="support-icon">✦</span>

            <strong>Learning for everyone</strong>

            <p>
              Assessment designed around your individual
              learning needs.
            </p>
          </div>

          <div className="profile">
            <div className="avatar">L</div>

            <div>
              <strong>Learner profile</strong>
              <span>Demo account</span>
            </div>

            <span className="profile-dots">•••</span>
          </div>
        </div>
      </aside>

      {/* MAIN CONTENT */}

      <main className="main-area">

        {/* TOP BAR */}

        <header className="topbar">
          <div className="breadcrumb">
            Workspace <span>/</span>{' '}

            {view === 'overview'
              ? 'Overview'
              : view === 'assessment'
                ? 'Assessment'
                : view === 'results'
                  ? 'Results'
                  : view === 'accessibility'
                    ? 'Accessibility'
                    : 'AI Prediction'}
          </div>

          <div className="topbar-actions">
            <span className="demo-badge">
              <span />
              API CONNECTED
            </span>

            <button
              className="access-shortcut"
              onClick={openAccessibility}
              aria-label="Open accessibility settings"
            >
              ◉
            </button>
          </div>
        </header>

        {/* GENERAL API ERROR */}

        {apiError && (
          <div className="api-error" role="alert">
            <strong>Notice:</strong> {apiError}

            <button
              onClick={() => setApiError('')}
              aria-label="Dismiss error"
            >
              ×
            </button>
          </div>
        )}

        {/* ====================================================
            OVERVIEW DASHBOARD
        ==================================================== */}

        {view === 'overview' && (
          <section className="page-content">
            <div className="welcome-row">
              <div>
                <div className="eyebrow">
                  YOUR LEARNING SPACE
                </div>

                <h1>Learn at your own pace.</h1>

                <p className="page-subtitle">
                  An inclusive assessment experience that
                  adapts to the way you learn.
                </p>
              </div>

              <div className="welcome-decoration">✳</div>
            </div>

            <section className="hero-card">
              <div className="hero-copy">
                <span className="hero-tag">
                  PERSONALISED LEARNING
                </span>

                <h2>
                  Your potential.
                  <br />
                  Your own pathway.
                </h2>

                <p>
                  Explore an adaptive assessment that adjusts
                  question difficulty based on your responses.
                </p>

                <button
                  className="primary-button"
                  onClick={startAssessment}
                  disabled={busy}
                >
                  {busy ? 'Connecting...' : 'Start assessment'}
                  <span>→</span>
                </button>
              </div>

              <div className="hero-art" aria-hidden="true">
                <div className="orbit orbit-one" />
                <div className="orbit orbit-two" />
                <div className="art-core">✦</div>
                <div className="art-dot dot-one" />
                <div className="art-dot dot-two" />
              </div>
            </section>

            <div className="section-heading">
              <div>
                <h2>Your overview</h2>
                <p>A snapshot of your learning activity.</p>
              </div>

              <span className="sample-label">
                LIVE API PROTOTYPE
              </span>
            </div>

            <div className="stats-grid">
              <article className="stat-card">
                <div className="stat-icon purple">▤</div>

                <span>Questions answered</span>

                <strong>{answeredCount}</strong>

                <small>In the current session</small>
              </article>

              <article className="stat-card">
                <div className="stat-icon green">✓</div>

                <span>Current accuracy</span>

                <strong>
                  {answeredCount
                    ? `${scorePercent}%`
                    : '—'}
                </strong>

                <small>Based on submitted answers</small>
              </article>

              <article className="stat-card">
                <div className="stat-icon amber">↗</div>

                <span>Learning level</span>

                <strong>
                  {sessionId ? ability : 'Not set'}
                </strong>

                <small>Updated by assessment engine</small>
              </article>
            </div>

            <div className="bottom-grid">
              <section className="panel">
                <div className="panel-heading">
                  <h3>Ready when you are</h3>
                  <span className="panel-icon">✦</span>
                </div>

                <p>
                  Begin an assessment and let the system adapt
                  as you answer.
                </p>

                <button
                  className="text-button"
                  onClick={startAssessment}
                  disabled={busy}
                >
                  Begin a new assessment →
                </button>
              </section>

              <section className="panel">
                <div className="panel-heading">
                  <h3>AI-powered insights</h3>
                  <span className="panel-icon">✦</span>
                </div>

                <p>
                  Explore the experimental Random Forest
                  educational prediction module.
                </p>

                <button
                  className="text-button"
                  onClick={() => setView('ai-prediction')}
                >
                  Explore AI prediction →
                </button>
              </section>
            </div>

            <p className="disclaimer">
              Prototype using a sample assessment question bank,
              rule-based adaptive logic and a separately trained
              student-performance prediction model.
            </p>
          </section>
        )}

        {/* ====================================================
            ADAPTIVE ASSESSMENT
        ==================================================== */}

        {view === 'assessment' && (
          <section className="page-content assessment-content">
            <div className="eyebrow">
              PERSONALISED ASSESSMENT
            </div>

            <h1>Show what you know.</h1>

            <p className="page-subtitle">
              Take your time. You can choose the answer
              that feels right to you.
            </p>

            {question ? (
              <>
                <div className="assessment-meta">
                  <span>
                    QUESTION {answeredCount + 1} OF 5
                  </span>

                  <span
                    className={`difficulty ${question.difficulty.toLowerCase()}`}
                  >
                    {question.difficulty} level
                  </span>
                </div>

                <div className="progress-track">
                  <div
                    style={{
                      width: `${(answeredCount / 5) * 100}%`,
                    }}
                  />
                </div>

                <article className="question-card">
                  <div className="question-top">
                    <span className="question-category">
                      {question.category}
                    </span>

                    <button
                      className="read-button"
                      onClick={readQuestion}
                    >
                      ♫ Read question aloud
                    </button>
                  </div>

                  <h2>{question.prompt}</h2>

                  <p className="question-hint">
                    Select one answer below.
                  </p>

                  <div className="options-list">
                    {question.options.map((option, index) => (
                      <button
                        key={`${question.id}-${index}`}
                        className={`option-button ${
                          selected === index ? 'selected' : ''
                        }`}
                        onClick={() =>
                          !feedback && setSelected(index)
                        }
                        disabled={!!feedback || busy}
                        aria-pressed={selected === index}
                      >
                        <span className="option-letter">
                          {String.fromCharCode(65 + index)}
                        </span>

                        <span>{option}</span>
                      </button>
                    ))}
                  </div>

                  {feedback && (
                    <div
                      className={`feedback-box ${
                        feedback.correct
                          ? 'feedback-correct'
                          : 'feedback-wrong'
                      }`}
                      role="status"
                    >
                      <strong>
                        {feedback.correct
                          ? 'Correct answer!'
                          : 'Not quite. Keep learning!'}
                      </strong>

                      <p>{feedback.explanation}</p>
                    </div>
                  )}

                  <div className="question-footer">
                    <span>
                      {feedback
                        ? 'Your response has been recorded by the backend.'
                        : 'Select an answer and submit when ready.'}
                    </span>

                    {!feedback ? (
                      <button
                        className="primary-button"
                        onClick={submitAnswer}
                        disabled={
                          selected === null || busy
                        }
                      >
                        {busy ? 'Submitting...' : 'Submit answer'}
                        <span>→</span>
                      </button>
                    ) : (
                      <button
                        className="primary-button"
                        onClick={nextQuestion}
                        disabled={busy}
                      >
                        {feedback.finished
                          ? 'View results'
                          : 'Next question'}
                        <span>→</span>
                      </button>
                    )}
                  </div>
                </article>
              </>
            ) : (
              <article className="question-card">
                <h2>No active assessment</h2>

                <p className="question-hint">
                  Start a new assessment to receive your first question.
                </p>

                <button
                  className="primary-button"
                  onClick={startAssessment}
                  disabled={busy}
                >
                  {busy ? 'Connecting...' : 'Start assessment'}
                  <span>→</span>
                </button>
              </article>
            )}

            <button
              className="back-link"
              onClick={() => setView('overview')}
            >
              ← Back to overview
            </button>
          </section>
        )}

        {/* ====================================================
            ACCESSIBILITY SETTINGS
        ==================================================== */}

        {view === 'accessibility' && (
          <section className="page-content">
            <div className="eyebrow">
              PERSONAL PREFERENCES
            </div>

            <h1>Accessibility settings.</h1>

            <p className="page-subtitle">
              Choose the display options that make learning
              more comfortable.
            </p>

            <div className="settings-panel">
              <div className="setting-row">
                <div>
                  <strong>Larger text</strong>
                  <p>
                    Increase the size of text throughout the application.
                  </p>
                </div>

                <button
                  className={`toggle ${
                    preferences.large_text ? 'on' : ''
                  }`}
                  onClick={() =>
                    setPreferences(previous => ({
                      ...previous,
                      large_text: !previous.large_text,
                    }))
                  }
                  aria-pressed={preferences.large_text}
                  aria-label="Toggle larger text"
                >
                  <span />
                </button>
              </div>

              <div className="setting-row">
                <div>
                  <strong>High contrast</strong>
                  <p>
                    Increase contrast between text,
                    backgrounds and controls.
                  </p>
                </div>

                <button
                  className={`toggle ${
                    preferences.high_contrast ? 'on' : ''
                  }`}
                  onClick={() =>
                    setPreferences(previous => ({
                      ...previous,
                      high_contrast: !previous.high_contrast,
                    }))
                  }
                  aria-pressed={preferences.high_contrast}
                  aria-label="Toggle high contrast"
                >
                  <span />
                </button>
              </div>

              <div className="setting-row">
                <div>
                  <strong>Reduce motion</strong>
                  <p>
                    Minimise interface transitions and animations.
                  </p>
                </div>

                <button
                  className={`toggle ${
                    preferences.reduced_motion ? 'on' : ''
                  }`}
                  onClick={() =>
                    setPreferences(previous => ({
                      ...previous,
                      reduced_motion: !previous.reduced_motion,
                    }))
                  }
                  aria-pressed={preferences.reduced_motion}
                  aria-label="Toggle reduced motion"
                >
                  <span />
                </button>
              </div>
            </div>

            <div className="access-note">
              <span>✦</span>

              <p>
                Preferences are stored through the backend for
                the current demo user.
              </p>
            </div>

            {settingsMessage && (
              <p className="success-message" role="status">
                {settingsMessage}
              </p>
            )}

            <button
              className="primary-button"
              onClick={savePreferences}
              disabled={busy}
            >
              {busy ? 'Saving...' : 'Save preferences'}
              <span>→</span>
            </button>
          </section>
        )}

        {/* ====================================================
            ASSESSMENT RESULTS
        ==================================================== */}

        {view === 'results' && (
          <section className="page-content">
            <div className="eyebrow">
              ASSESSMENT SUMMARY
            </div>

            <h1>Your progress, your journey.</h1>

            <p className="page-subtitle">
              Here is the summary returned by your assessment backend.
            </p>

            {results ? (
              <div className="result-card">
                <div className="result-score">
                  {Math.round(results.accuracy)}
                  <span>%</span>
                </div>

                <h2>
                  {results.accuracy >= 80
                    ? 'Great work!'
                    : results.accuracy >= 50
                      ? 'Good effort!'
                      : 'Every attempt helps you grow.'}
                </h2>

                <p>
                  You answered {results.score} out of{' '}
                  {results.answered_count} questions correctly.
                </p>

                <div className="result-details">
                  <div>
                    <span>Questions answered</span>
                    <strong>{results.answered_count}</strong>
                  </div>

                  <div>
                    <span>Correct answers</span>
                    <strong>{results.score}</strong>
                  </div>

                  <div>
                    <span>Final difficulty level</span>
                    <strong>{results.ability_level}</strong>
                  </div>
                </div>

                <button
                  className="primary-button"
                  onClick={startAssessment}
                  disabled={busy}
                >
                  Try another assessment <span>→</span>
                </button>
              </div>
            ) : (
              <div className="result-card">
                <div className="result-score">—</div>

                <h2>No completed assessment yet.</h2>

                <p>
                  Complete an assessment to see your score
                  and learning summary here.
                </p>

                <button
                  className="primary-button"
                  onClick={startAssessment}
                  disabled={busy}
                >
                  Start assessment <span>→</span>
                </button>
              </div>
            )}

            <button
              className="back-link"
              onClick={() => setView('overview')}
            >
              ← Back to overview
            </button>
          </section>
        )}

        {/* ====================================================
            NEW RANDOM FOREST AI PREDICTION PAGE
        ==================================================== */}

        {view === 'ai-prediction' && (
          <StudentRiskPrediction />
        )}

        {/* FOOTER */}

        <footer className="app-footer">
          <span>
            INCLUSIFY · AI INCLUSIVE ASSESSMENT SYSTEM
          </span>

          <span>
            API-connected prototype · Experimental ML module
          </span>
        </footer>
      </main>
    </div>
  )
}

export default App
