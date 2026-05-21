/**
 * Video Upload + Results Component
 * Full two-column layout: controls left, output right, violations gallery below
 */

import { useState, useRef, useCallback } from 'react'
import toast from 'react-hot-toast'
import api, { runPipeline, runPipelineLocalPath } from '../api/client'

const ALLOWED_TYPES = ['video/mp4', 'video/avi', 'video/quicktime', 'video/webm', 'video/x-matroska']
const MAX_SIZE = Infinity

// Convert any backend path to a proxy-able URL
function toImgUrl(path) {
    if (!path) return null
    if (path.startsWith('/uploads/') || path.startsWith('/reports/')) return path
    const norm = path.replace(/\\/g, '/')
    const idx = norm.lastIndexOf('/uploads/')
    if (idx !== -1) return norm.slice(idx)
    return `/uploads/${norm.split('/').pop()}`
}

// Badge colors per violation type
const BADGE = {
    no_helmet: { bg: '#f59e0b', label: 'No Helmet' },
    no_vest: { bg: '#f59e0b', label: 'No Vest' },
    both_missing: { bg: '#ef4444', label: 'Both Missing' },
}

function VideoUpload({ onResultChange }) {
    const [mode, setMode] = useState('upload') // 'upload' | 'local'
    const [selectedFile, setSelectedFile] = useState(null)
    const [localPath, setLocalPath] = useState('')
    const [isProcessing, setIsProcessing] = useState(false)
    const [progress, setProgress] = useState(0)
    const [progressText, setProgressText] = useState('')
    const [result, setResult] = useState(null)
    const [frameSkip, setFrameSkip] = useState(5)
    const fileInputRef = useRef(null)

    const handleFileSelect = useCallback((file) => {
        if (!file) return
        if (!ALLOWED_TYPES.includes(file.type)) {
            toast.error('Invalid file type. Use MP4, AVI, MOV, WEBM, or MKV')
            return
        }
        setSelectedFile(file)
        setResult(null)
        onResultChange?.(null)
    }, [onResultChange])

    const handleDrop = useCallback((e) => {
        e.preventDefault()
        e.stopPropagation()
        handleFileSelect(e.dataTransfer.files[0])
    }, [handleFileSelect])

    const handleProcess = async () => {
        if (mode === 'upload' && !selectedFile) { toast.error('Please select a video first'); return }
        if (mode === 'local' && !localPath.trim()) { toast.error('Please enter a server file path'); return }

        setIsProcessing(true)
        setProgress(0)
        setResult(null)
        onResultChange?.(null)

        try {
            const loadingToast = toast.loading('Running Sentry-Judge pipeline...')

            let data
            if (mode === 'local') {
                setProgressText('Reading file from server...')
                setProgress(15)
                setProgressText('Sentry processing (YOLO + ByteTrack)...')
                setProgress(35)
                data = await runPipelineLocalPath(localPath.trim())
            } else {
                setProgressText('Uploading video...')
                setProgress(10)
                setProgressText('Sentry processing (YOLO + ByteTrack)...')
                setProgress(30)
                data = await runPipeline(selectedFile)
            }

            setProgressText('Judge verification complete!')
            setProgress(100)
            setResult(data)
            onResultChange?.(data)

            const confirmed = data.judge?.confirmed || 0
            toast.success(
                `Pipeline done! ${confirmed} verified violations · ${data.sentry?.effective_fps} FPS`,
                { id: loadingToast }
            )
        } catch (error) {
            console.error('Video processing error:', error)
            toast.error(error.response?.data?.detail || 'Video processing failed')
        } finally {
            setIsProcessing(false)
            setProgressText('')
        }
    }

    const handleClear = () => {
        setSelectedFile(null)
        setLocalPath('')
        setResult(null)
        onResultChange?.(null)
        setProgress(0)
        setProgressText('')
        if (fileInputRef.current) fileInputRef.current.value = ''
    }

    const { sentry, judge, bypass_rate, verified_violations = [], output_video_url } = result || {}

    // Ensure video URL goes through the vite proxy (strip absolute origin if present)
    const videoUrl = output_video_url
        ? (output_video_url.startsWith('http')
            ? '/' + output_video_url.split('/').slice(3).join('/')
            : output_video_url)
        : null

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>

            {/* ── Top row: controls left / video right ─────────────────── */}
            <div style={{ display: 'grid', gridTemplateColumns: '320px 1fr', gap: '1.5rem', alignItems: 'start' }}>

                {/* LEFT: Controls */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>

                    {/* Mode toggle */}
                    <div style={{ display: 'flex', background: 'var(--bg-secondary)', borderRadius: 8, padding: 3, gap: 3 }}>
                        {[['upload', '📤 Upload File'], ['local', '📂 Server Path']].map(([m, label]) => (
                            <button
                                key={m}
                                onClick={() => { setMode(m); setResult(null); onResultChange?.(null) }}
                                disabled={isProcessing}
                                style={{
                                    flex: 1, padding: '0.4rem 0.5rem', borderRadius: 6,
                                    border: 'none', cursor: 'pointer', fontSize: '0.8rem', fontWeight: 600,
                                    background: mode === m ? 'var(--bg-primary)' : 'transparent',
                                    color: mode === m ? 'var(--text-primary)' : 'var(--text-muted)',
                                    boxShadow: mode === m ? '0 1px 3px rgba(0,0,0,0.2)' : 'none',
                                    transition: 'all 0.15s',
                                }}
                            >
                                {label}
                            </button>
                        ))}
                    </div>

                    {/* Drop zone (upload mode) */}
                    {mode === 'local' ? (
                        <div className="card" style={{ padding: '1rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                            <label style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)' }}>
                                Absolute path on the server
                            </label>
                            <input
                                type="text"
                                value={localPath}
                                onChange={e => setLocalPath(e.target.value)}
                                placeholder="/content/video.mp4"
                                disabled={isProcessing}
                                style={{
                                    width: '100%', padding: '0.5rem 0.75rem',
                                    borderRadius: 6, border: '1px solid var(--border-color)',
                                    background: 'var(--bg-primary)', color: 'var(--text-primary)',
                                    fontFamily: 'monospace', fontSize: '0.85rem',
                                    boxSizing: 'border-box',
                                }}
                            />
                            <small style={{ color: 'var(--text-muted)', fontSize: '0.75rem' }}>
                                Upload the video directly to Colab via <code>files.upload()</code>, then paste the returned path here to bypass ngrok.
                            </small>
                        </div>
                    ) : (
                    <div
                        className={`upload-zone ${selectedFile ? 'upload-zone--has-file' : ''}`}
                        style={{ minHeight: 160 }}
                        onDrop={handleDrop}
                        onDragOver={(e) => e.preventDefault()}
                        onClick={() => fileInputRef.current?.click()}
                    >
                        <input
                            ref={fileInputRef}
                            type="file"
                            accept=".mp4,.avi,.mov,.webm,.mkv"
                            style={{ display: 'none' }}
                            onChange={(e) => handleFileSelect(e.target.files[0])}
                        />
                        {selectedFile ? (
                            <div className="upload-zone__preview">
                                <span style={{ fontSize: '2rem' }}>🎬</span>
                                <p style={{ fontWeight: 600, margin: '0.25rem 0' }}>{selectedFile.name}</p>
                                <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>
                                    {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                                </p>
                            </div>
                        ) : (
                            <div className="upload-zone__placeholder">
                                <span style={{ fontSize: '2.5rem' }}>📹</span>
                                <p style={{ margin: '0.5rem 0 0.25rem' }}>Drop video or click to select</p>
                                <small style={{ color: 'var(--text-muted)' }}>MP4, AVI, MOV, WEBM · max 100 MB</small>
                            </div>
                        )}
                    </div>
                    )}

                    {/* Pipeline info */}
                    <div className="card" style={{ padding: '0.75rem' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                            <span style={{ fontSize: '1.1rem' }}>🛡️</span>
                            <div>
                                <div style={{ fontWeight: 600, fontSize: '0.9rem' }}>Sentry-Judge Pipeline</div>
                                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                                    YOLO → Queue → SAM 3 verification → DB
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Progress */}
                    {isProcessing && (
                        <div className="progress-container">
                            <div className="progress-bar">
                                <div className="progress-bar__fill" style={{ width: `${progress}%` }} />
                            </div>
                            <span className="progress-text">{progressText || `${progress}%`}</span>
                        </div>
                    )}

                    {/* Buttons */}
                    <div style={{ display: 'flex', gap: '0.75rem' }}>
                        <button
                            className="btn btn--primary btn--lg"
                            style={{ flex: 1 }}
                            onClick={handleProcess}
                            disabled={isProcessing || (mode === 'upload' ? !selectedFile : !localPath.trim())}
                        >
                            {isProcessing ? (
                                <><span className="loading-spinner" style={{ width: 16, height: 16 }} /> Running...</>
                            ) : '🛡️ Run Pipeline'}
                        </button>
                        {(selectedFile || localPath) && (
                            <button className="btn btn--secondary btn--lg" onClick={handleClear} disabled={isProcessing}>
                                Clear
                            </button>
                        )}
                    </div>

                    {/* Stats summary cards (post-result) */}
                    {result && sentry && (
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem' }}>
                            {[
                                { label: 'FPS', value: sentry.effective_fps, color: 'var(--color-safe)' },
                                { label: 'Tracked', value: sentry.unique_persons },
                                { label: 'Confirmed', value: judge?.confirmed, color: judge?.confirmed > 0 ? 'var(--color-violation)' : undefined },
                                { label: 'Rejected', value: judge?.rejected, color: 'var(--color-safe)' },
                                { label: 'SAM Bypass', value: `${bypass_rate}%`, color: 'var(--color-safe)' },
                                { label: 'Confirm Rate', value: `${judge?.confirmation_rate}%` },
                            ].map(({ label, value, color }) => (
                                <div key={label} className="card" style={{ padding: '0.6rem', textAlign: 'center' }}>
                                    <div style={{ fontSize: '1.25rem', fontWeight: 700, color: color || 'var(--text-primary)' }}>
                                        {value ?? '-'}
                                    </div>
                                    <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: 2 }}>{label}</div>
                                </div>
                            ))}
                        </div>
                    )}

                    {/* Decision paths */}
                    {sentry?.path_distribution && Object.keys(sentry.path_distribution).length > 0 && (
                        <div className="card" style={{ padding: '0.75rem' }}>
                            <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '0.5rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                                🔀 Decision Paths
                            </div>
                            {Object.entries(sentry.path_distribution).map(([path, count]) => (
                                <div key={path} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', marginBottom: '0.25rem' }}>
                                    <span>{path}</span>
                                    <span className="badge badge--info">{count}</span>
                                </div>
                            ))}
                        </div>
                    )}
                </div>

                {/* RIGHT: Video output or placeholder */}
                <div>
                    {videoUrl ? (
                        <div className="card" style={{ padding: '1rem' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
                                <h3 style={{ margin: 0 }}>🎬 Annotated Output</h3>
                                <a href={videoUrl} download className="btn btn--secondary" style={{ fontSize: '0.8rem' }}>
                                    📥 Download
                                </a>
                            </div>
                            <video
                                key={videoUrl}
                                controls
                                style={{ width: '100%', borderRadius: 8, background: '#000', maxHeight: 480 }}
                            >
                                <source src={videoUrl} type="video/mp4" />
                                Your browser does not support video playback.
                            </video>
                        </div>
                    ) : (
                        <div className="card" style={{ textAlign: 'center', padding: '4rem 2rem', height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: 320 }}>
                            {isProcessing ? (
                                <>
                                    <span className="loading-spinner" style={{ width: 48, height: 48, marginBottom: '1rem' }} />
                                    <h3 style={{ marginBottom: '0.5rem' }}>Processing Video...</h3>
                                    <p style={{ color: 'var(--text-muted)' }}>{progressText}</p>
                                </>
                            ) : (
                                <>
                                    <div style={{ fontSize: '4rem', opacity: 0.2, marginBottom: '1rem' }}>🎬</div>
                                    <h3 style={{ marginBottom: '0.5rem' }}>Video PPE Detection</h3>
                                    <p style={{ color: 'var(--text-muted)', maxWidth: 380 }}>
                                        Upload a video to run the Sentry-Judge pipeline. The annotated output will appear here.
                                    </p>
                                    <div style={{ display: 'flex', gap: '1.5rem', marginTop: '2rem', justifyContent: 'center' }}>
                                        {[['📹', 'MP4, AVI, MOV'], ['⚡', 'GPU Accelerated'], ['📊', 'Aggregate Stats']].map(([icon, label]) => (
                                            <div key={label} style={{ textAlign: 'center' }}>
                                                <div style={{ fontSize: '1.8rem' }}>{icon}</div>
                                                <small style={{ color: 'var(--text-muted)' }}>{label}</small>
                                            </div>
                                        ))}
                                    </div>
                                </>
                            )}
                        </div>
                    )}
                </div>
            </div>

            {/* ── Verified Violations Gallery ───────────────────────────── */}
            {verified_violations.length > 0 && (
                <div className="card" style={{ padding: '1.25rem' }}>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
                        <h3 style={{ margin: 0 }}>⚖️ Judge-Confirmed Violations</h3>
                        <div style={{ display: 'flex', gap: '0.75rem' }}>
                            <span style={{ background: '#ef444420', color: '#ef4444', padding: '0.25rem 0.75rem', borderRadius: 20, fontSize: '0.8rem', fontWeight: 600 }}>
                                {judge?.confirmed} Confirmed
                            </span>
                            <span style={{ background: '#10b98120', color: '#10b981', padding: '0.25rem 0.75rem', borderRadius: 20, fontSize: '0.8rem', fontWeight: 600 }}>
                                {judge?.rejected} Rejected
                            </span>
                        </div>
                    </div>
                    <div style={{
                        display: 'grid',
                        gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))',
                        gap: '1rem'
                    }}>
                        {verified_violations.map((v, idx) => {
                            const imgUrl = toImgUrl(v.image_path || v.cropped_roi_path)
                            const badge = BADGE[v.violation_type] || { bg: '#6b7280', label: v.violation_type }
                            return (
                                <div key={v.id || idx} style={{
                                    background: 'var(--bg-secondary)',
                                    borderRadius: 10,
                                    overflow: 'hidden',
                                    border: '1px solid var(--border-color)',
                                    transition: 'transform 0.15s',
                                }}
                                    onMouseEnter={e => e.currentTarget.style.transform = 'translateY(-2px)'}
                                    onMouseLeave={e => e.currentTarget.style.transform = 'none'}
                                >
                                    {/* ROI image */}
                                    <div style={{ height: 140, background: '#111', position: 'relative', overflow: 'hidden' }}>
                                        {imgUrl ? (
                                            <img
                                                src={imgUrl}
                                                alt="Evidence ROI"
                                                style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                                                onError={e => { e.target.parentElement.innerHTML = '<div style="height:100%;display:flex;align-items:center;justify-content:center;font-size:2rem;opacity:0.3">👷</div>' }}
                                            />
                                        ) : (
                                            <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '2.5rem', opacity: 0.2 }}>👷</div>
                                        )}
                                        {/* Violation type badge overlay */}
                                        <div style={{
                                            position: 'absolute', bottom: 6, left: 6,
                                            background: badge.bg, color: '#fff',
                                            padding: '2px 8px', borderRadius: 12,
                                            fontSize: '0.7rem', fontWeight: 700
                                        }}>
                                            {badge.label}
                                        </div>
                                    </div>

                                    {/* Details */}
                                    <div style={{ padding: '0.6rem' }}>
                                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                                            <span style={{ fontWeight: 600, fontSize: '0.85rem' }}>
                                                Person {v.person_id}
                                            </span>
                                            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                                                {v.camera_zone || 'CAM-001'}
                                            </span>
                                        </div>
                                        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                                            Conf: <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>
                                                {v.judge_confidence?.toFixed(3) ?? '-'}
                                            </span>
                                        </div>
                                        {v.timestamp && (
                                            <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: 2 }}>
                                                {new Date(v.timestamp).toLocaleTimeString()}
                                            </div>
                                        )}
                                    </div>
                                </div>
                            )
                        })}
                    </div>
                </div>
            )}
        </div>
    )
}

export default VideoUpload
