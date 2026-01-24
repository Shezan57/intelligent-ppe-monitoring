/**
 * StatsPanel Component
 * Displays detection statistics and metrics
 */

const StatsPanel = ({ stats, timing }) => {
    if (!stats) {
        return (
            <div className="card">
                <div className="card__header">
                    <h3 className="card__title">📊 Statistics</h3>
                </div>
                <div className="stats-panel">
                    <div className="stat-item">
                        <div className="stat-item__value">-</div>
                        <div className="stat-item__label">Persons</div>
                    </div>
                    <div className="stat-item">
                        <div className="stat-item__value">-</div>
                        <div className="stat-item__label">Violations</div>
                    </div>
                    <div className="stat-item">
                        <div className="stat-item__value">-</div>
                        <div className="stat-item__label">Compliance</div>
                    </div>
                    <div className="stat-item">
                        <div className="stat-item__value">-</div>
                        <div className="stat-item__label">SAM Calls</div>
                    </div>
                </div>
            </div>
        )
    }

    const complianceColor = stats.compliance_rate >= 80 ? 'safe' : 'violation'

    return (
        <div className="card">
            <div className="card__header">
                <h3 className="card__title">📊 Statistics</h3>
            </div>

            <div className="stats-panel">
                <div className="stat-item">
                    <div className="stat-item__value">{stats.total_persons}</div>
                    <div className="stat-item__label">Persons</div>
                </div>

                <div className={`stat-item stat-item--${stats.total_violations > 0 ? 'violation' : 'safe'}`}>
                    <div className="stat-item__value">{stats.total_violations}</div>
                    <div className="stat-item__label">Violations</div>
                </div>

                <div className={`stat-item stat-item--${complianceColor}`}>
                    <div className="stat-item__value">{stats.compliance_rate.toFixed(0)}%</div>
                    <div className="stat-item__label">Compliance</div>
                </div>

                <div className="stat-item">
                    <div className="stat-item__value">{stats.sam_activations}</div>
                    <div className="stat-item__label">SAM Calls</div>
                </div>
            </div>

            {/* Timing breakdown */}
            {timing && (
                <div className="mt-md">
                    <h4 style={{ fontSize: '0.875rem', marginBottom: '0.5rem', color: 'var(--text-secondary)' }}>
                        Processing Time
                    </h4>
                    <div className="progress-bar">
                        <div
                            className="progress-bar__fill"
                            style={{ width: `${Math.min((timing.yolo_ms / timing.total_ms) * 100, 100)}%` }}
                        ></div>
                    </div>
                    <div className="flex justify-between mt-sm" style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                        <span>YOLO: {timing.yolo_ms.toFixed(1)}ms</span>
                        <span>SAM: {timing.sam_ms.toFixed(1)}ms</span>
                        <span>Total: {timing.total_ms.toFixed(1)}ms</span>
                    </div>
                </div>
            )}

            {/* Path distribution */}
            {stats.path_distribution && (
                <div className="mt-md">
                    <h4 style={{ fontSize: '0.875rem', marginBottom: '0.5rem', color: 'var(--text-secondary)' }}>
                        Decision Paths
                    </h4>
                    {Object.entries(stats.path_distribution).map(([path, count]) => (
                        <div key={path} className="flex justify-between mb-sm" style={{ fontSize: '0.8rem' }}>
                            <span>{path}</span>
                            <span className="badge badge--info">{count}</span>
                        </div>
                    ))}
                </div>
            )}
        </div>
    )
}

export default StatsPanel
