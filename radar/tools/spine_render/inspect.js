// 诊断：把 skeleton 的 slot/附件 和 atlas 的 region 名对照出来，看差在哪。
//
// 用法：node inspect.js <atlas> <skel>

const fs = require("fs");
const path = require("path");
const vm = require("vm");
const { createCanvas } = require("@napi-rs/canvas");

function loadSpine() {
  const sandbox = {
    console, Math, Date, JSON, Object, Array, String, Number, Boolean, Error,
    Float32Array, Uint16Array, Uint32Array, Int32Array, Uint8Array, ArrayBuffer,
    performance: { now: () => Date.now() },
    window: {},
  };
  sandbox.globalThis = sandbox;
  vm.createContext(sandbox);
  vm.runInContext("var spine = {};", sandbox);
  for (const file of ["spine-core.js", "spine-canvas.js"]) {
    vm.runInContext(fs.readFileSync(path.join(__dirname, file), "utf8"), sandbox, { filename: file });
  }
  return sandbox.spine;
}

const spine = loadSpine();

function main() {
  const [atlasPath, skelPath] = process.argv.slice(2);
  const dir = path.dirname(atlasPath);
  const atlasText = fs.readFileSync(atlasPath, "utf8");
  const atlas = new spine.TextureAtlas(atlasText, () => new spine.canvas.CanvasTexture(createCanvas(256, 256)));

  const binary = new spine.SkeletonBinary(new spine.AtlasAttachmentLoader(atlas));
  const data = binary.readSkeletonData(new Uint8Array(fs.readFileSync(skelPath)));
  const skeleton = new spine.Skeleton(data);

  console.log("== atlas regions ==");
  console.log(atlas.regions.map((r) => `${r.name}(${r.width}x${r.height})`).join(", "));
  console.log("\n== skins ==");
  console.log(data.skins.map((s) => s.name).join(", "));
  console.log("\n== slots / attachments ==");
  for (const slot of skeleton.slots) {
    const attachment = slot.getAttachment();
    if (!attachment) continue;
    const regionName = attachment.region ? attachment.region.name : "(null)";
    console.log(
      `${slot.data.name.padEnd(22)} att=${String(attachment.name).padEnd(22)} type=${attachment.constructor.name.padEnd(16)} region=${regionName}`
    );
  }
  console.log("\n== animations ==");
  console.log(data.animations.map((a) => a.name).join(", "));
}

main();
