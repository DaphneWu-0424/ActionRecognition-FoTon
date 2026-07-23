import { NavLink, Outlet } from "react-router-dom";

export function AppShell() {
  return (
    <div className="app-shell">
      <header className="topbar">
        <NavLink to="/" className="brand">
          <span className="brand-mark">AF</span>
          <span>
            <strong>Action Finder</strong>
            <small>LOCAL VISUAL SEARCH</small>
          </span>
        </NavLink>
        <nav>
          <NavLink to="/assets">媒体库</NavLink>
          <NavLink to="/jobs/new">新建检索</NavLink>
          <NavLink to="/jobs">任务记录</NavLink>
        </nav>
        <div className="local-chip">
          <span />
          LOCAL · CPU
        </div>
      </header>
      <main>
        <Outlet />
      </main>
    </div>
  );
}
