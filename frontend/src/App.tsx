import { Navigate, Route, Routes } from "react-router-dom";
import { AppShell } from "./components/AppShell";
import { AssetLibrary } from "./pages/AssetLibrary";
import { CreateJob } from "./pages/CreateJob";
import { JobDetail } from "./pages/JobDetail";
import { JobList } from "./pages/JobList";

export default function App() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route index element={<Navigate to="/assets" replace />} />
        <Route path="/assets" element={<AssetLibrary />} />
        <Route path="/jobs/new" element={<CreateJob />} />
        <Route path="/jobs" element={<JobList />} />
        <Route path="/jobs/:jobId" element={<JobDetail />} />
      </Route>
    </Routes>
  );
}
