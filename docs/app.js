"use strict";
const scenes = [
  {id:"office_chair",title:"Office Chair",task:"Reach the office chair farthest from the trash can.",avoid:"The table nearest the starting position.",failure:"Baseline falls; no table contact.",duration:"8.67 s",phase:"30%"},
  {id:"bedroom_door",title:"Bedroom Door",task:"Reach the door far from the armchair.",avoid:"The bed. No body part may touch it.",failure:"Baseline falls; no bed contact.",duration:"9.27 s",phase:"30%"},
  {id:"lounge_armchair",title:"Lounge Armchair",task:"Reach the armchair near the shelf.",avoid:"The coffee table. No body part may touch it.",failure:"Baseline left hand contacts the coffee table.",duration:"9.07 s",phase:"60%"}
];
const modes = [
  {id:"mesh",title:"Original Motion",subtitle:"Mesh · kinematic reference"},
  {id:"baseline",title:"Baseline Controller",subtitle:"Capsule Physics · recorded failure"},
  {id:"h1",title:"H1 Model",subtitle:"Capsule Physics · successful recovery"}
];
const root = document.getElementById("scenes");
for (const s of scenes) {
  const section = document.createElement("section");
  section.className = "scene"; section.id = s.id;
  section.setAttribute("aria-labelledby", `${s.id}-title`);
  section.innerHTML = `
    <div class="scene-heading">
      <div><h2 id="${s.id}-title">${s.title}</h2><p><strong>Task</strong> ${s.task}</p><p><strong>Avoid</strong> ${s.avoid}</p></div>
      <div class="scene-result"><span class="success">H1: goal reached</span><p>${s.failure}</p></div>
    </div>
    <video class="comparison" controls playsinline preload="metadata" poster="media/${s.id}/poster.jpg" aria-label="${s.title}: Mesh, baseline and H1 comparison">
      <source src="media/${s.id}/comparison.mp4" type="video/mp4">
      <a href="media/${s.id}/comparison.mp4">Download the ${s.title} comparison</a>
    </video>
    <div class="video-footer"><span>${s.duration} · Complete recordings at 0.5× · Push at ${s.phase} of the reference motion</span><a href="media/${s.id}/comparison.mp4" download>Download comparison MP4</a></div>
    <details><summary>View the three individual videos</summary><div class="individuals">${modes.map(m => `
      <article class="clip"><h3>${m.title}</h3><p>${m.subtitle}</p>
      <video controls playsinline preload="none" aria-label="${s.title}: ${m.title}">
      <source src="media/${s.id}/${m.id}.mp4" type="video/mp4"></video>
      <a href="media/${s.id}/${m.id}.mp4" download>Download MP4</a></article>`).join("")}</div></details>`;
  root.appendChild(section);
}
// Keep a single active playback so comparisons do not compete for attention.
document.addEventListener("play", event => {
  if (!(event.target instanceof HTMLVideoElement)) return;
  document.querySelectorAll("video").forEach(video => {
    if (video !== event.target) video.pause();
  });
}, true);
