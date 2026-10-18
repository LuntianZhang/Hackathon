const {readFileSync} = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const test = require('node:test');
const assert = require('node:assert/strict');

function app() {
  const location = {hash: '#scan/photo'};
  const history = {state: null};
  for (const method of ['pushState', 'replaceState']) history[method] = (value, _, url) => {
    history.state = value; location.hash = url;
  };
  const scans = {sessions: [{id: 'scan123'}], scan: null, configure() {}, leave() {},
    async enter() {}, async refresh() {}};
  const context = vm.createContext({document: {addEventListener() {}, getElementById() {return {addEventListener() {}};}},
    window: {addEventListener() {}}, navigator: {}, phoneScan: scans,
    location, history, Intl, Date, URL, clearTimeout, setTimeout});
  const source = readFileSync(path.join(__dirname, '../../app/web/app.js'), 'utf8')
    .replace(/^import[^\n]*\n/, '').replace(/^await bootstrap\(\);\r?$/m, '');
  vm.runInContext(source, context);
  vm.runInContext(`render=()=>{}; refreshRecords=async()=>{}; pollMeal=()=>{};
    api=async path=>({id:path.split('/')[2],items:[],totals:{},saved_at:'saved'});`, context);
  return {context, state: vm.runInContext('state', context), scans, location, history,
    run: code => vm.runInContext(code, context)};
}

test('refresh and browser history restore record branch and selected scan or plate', async () => {
  for (const [hash, flow, selected] of [
    ['#scan/video', 'video'], ['#scan/mobile/scan123', 'mobile', 'scan123'],
    ['#scan/edit/meal123', 'edit', 'meal123']]) {
    const a = app(); a.location.hash = hash;
    await a.run('route()');
    assert.equal(a.state.page, 'scan'); assert.equal(a.state.captureFlow, flow);
    assert.equal(a.location.hash, hash);
    if (flow === 'mobile') assert.equal(a.scans.scan.id, selected);
    if (flow === 'edit') assert.equal(a.state.historyMealId, selected);
  }
});

test('opening a saved meal from scanning preserves its origin through refresh and 3D', async () => {
  const a = app(); a.state.page = 'scan'; a.state.captureFlow = 'mobile'; a.scans.scan = a.scans.sessions[0];
  await a.run("openMeal('meal123')");
  assert.equal(a.history.state.mealReturn.flow, 'mobile');
  assert.equal(a.state.mealReturn.selection, 'scan123');
  a.state.mealReturn = null; // simulate reloading the application
  await a.run('route()');
  assert.equal(a.state.mealReturn.flow, 'mobile');
  a.state.meal.three_d_available = true;
  await a.run("go('plate')");
  assert.equal(a.state.mealReturn.flow, 'mobile');
  await a.run("action('return-record')");
  assert.equal(a.location.hash, '#scan/mobile/scan123');
});

test('unavailable 3D falls back to result and the result omits the model button', async () => {
  const a = app();
  a.state.meal = {id: 'manual', source_kind: 'manual', title: 'Meal',
    occurred_at: new Date().toISOString(), meal_type: 'snack', items: [], totals: {}, analysis: {status: 'ready'}};
  await a.run("go('plate')");
  assert.equal(a.state.page, 'result');
  const html = a.run('result()');
  assert.ok(!html.includes('data-mode="3d"'));
});

test('opening the same meal from the diary clears an old scanning return target', async () => {
  const a = app(); a.state.page = 'diary';
  a.state.mealReturn = {mealId:'meal123',flow:'mobile',selection:'scan123'};
  a.state.nutritionReturnMealId = 'meal123';
  await a.run("openMeal('meal123')");
  assert.equal(a.state.mealReturn, null);
  assert.equal(a.state.nutritionReturnMealId, null);
});

test('returning to an existing plate restores that plate after viewing another meal', async () => {
  const a = app(); a.state.page = 'result'; a.state.captureFlow = 'edit';
  a.state.recognitionParent = {id:'original'};
  a.state.meal = {id:'other'};
  await a.run("go('scan')");
  assert.equal(a.state.meal.id, 'original');
  assert.equal(a.location.hash, '#scan/edit/original');
});

test('nutrition descriptions reflect actual portion methods and excluded object kinds', () => {
  const a = app();
  assert.match(a.run("quantityNote({volume_density_items:1})"), /体积与密度/);
  assert.match(a.run("quantityNote({volume_density_items:1,reference_items:1})"), /两种/);
  assert.match(a.run("quantityNote({quantity_pending_items:1})"), /暂不计入/);
  assert.match(a.run("foodNutritionLabel({object_kind:'non_food'})"), /不计入营养/);
  assert.match(a.run("foodNutritionLabel({object_kind:'packaged_food'})"), /不据此估算营养/);
});

test('direct list editing describes changes to the original record', () => {
  const a = app(); a.state.captureFlow = 'edit'; a.state.assistMode = 'none';
  a.state.recognitionParent = {id:'meal123',title:'Original',threeD:false};
  a.state.file = {name:'reference.jpg'};
  const html = a.run('scan()');
  assert.match(html, /直接修改原记录的食物清单，不生成新记录/);
  assert.ok(!html.includes('重新识别会生成新草稿'));
});
